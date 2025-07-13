# SPDX-FileCopyrightText: 2025 Mohamed Khayat
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from typing import List, Tuple, Dict, Any
import torch
from torch.amp import GradScaler
from tqdm import tqdm
from transformers.image_transforms import center_to_corners_format
from omegaconf import DictConfig
from fruit_project.utils.datasets.alb_mosaic_dataset import AlbumentationsMosaicDataset
from fruit_project.utils.early_stop import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from fruit_project.utils.metrics import ConfusionMatrix, MAPEvaluator
from transformers import AutoImageProcessor, BatchEncoding
import torch.nn as nn
from wandb.sdk.wandb_run import Run
from torch.utils.data import DataLoader
from torch.optim import AdamW, Optimizer
from fruit_project.utils.logging import (
    log_checkpoint_artifact,
    log_epoch_data,
    log_val_data,
)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        processor: AutoImageProcessor,
        device: torch.device,
        cfg: DictConfig,
        name: str,
        run: Run,
        train_dl: DataLoader,
        test_dl: DataLoader,
        val_dl: DataLoader,
    ):
        """
        Initializes the Trainer object.

        Args:
            model (nn.Module): The model to train.
            processor (AutoImageProcessor): The processor for preprocessing data.
            device (torch.device): The device to run training on.
            cfg (DictConfig): The configuration object.
            name (str): The name of the training run.
            run (Run): The wandb run object.
            train_dl (DataLoader): The training dataloader.
            test_dl (DataLoader): The testing dataloader.
            val_dl (DataLoader): The validation dataloader.
        """
        self.model: nn.Module = model
        self.device: torch.device = device
        self.scaler = GradScaler("cuda")
        self.cfg: DictConfig = cfg
        self.optimizer: Optimizer = self.get_optimizer()
        self.processor: AutoImageProcessor = processor
        self.name: str = name
        self.early_stopping: EarlyStopping = EarlyStopping(
            cfg.patience, cfg.delta, "checkpoints", name, run, cfg.log, cfg.upload
        )
        self.scheduler: SequentialLR = self.get_scheduler()
        self.run: Run = run
        self.train_dl: DataLoader = train_dl
        self.test_dl: DataLoader = test_dl
        self.val_dl: DataLoader = val_dl
        self.start_epoch: int = 0
        self.map_evaluator = MAPEvaluator(
            image_processor=processor,
            device=self.device,
            threshold=0.01,
            id2label=train_dl.dataset.id2lbl,
        )
        self.accum_steps: int = (
            self.cfg.effective_batch_size // self.cfg.step_batch_size
        )
        assert self.cfg.effective_batch_size % self.cfg.step_batch_size == 0, (
            f"effective_batch_size ({self.cfg.effective_batch_size}) must be divisible by batch_size "
            f"({self.accum_steps})."
        )

    def get_scheduler(self) -> SequentialLR:
        """
        Creates a learning rate scheduler with a warmup phase.

        Returns:
            SequentialLR: The learning rate scheduler.
        """
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, total_iters=self.cfg.warmup_epochs
        )

        main_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.epochs - self.cfg.warmup_epochs, eta_min=1e-7
        )

        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.cfg.warmup_epochs],
        )
        return scheduler

    def get_optimizer(self) -> AdamW:
        """
        Creates an AdamW optimizer with different learning rates for backbone and other parameters.

        Returns:
            AdamW: The optimizer.
        """
        # Backbone (pre-trained - lowest LR)
        backbone_params = [
            p
            for n, p in self.model.named_parameters()
            if "backbone" in n and p.requires_grad
        ]

        # Encoder/Decoder (pre-trained but task-specific - medium LR)
        encoder_decoder_params = [
            p
            for n, p in self.model.named_parameters()
            if any(
                head in n
                for head in [
                    "encoder.encoder",
                    "decoder.layers",
                    "input_proj",
                    "enc_output",
                ]
            )  # Target the main transformer layers
            and p.requires_grad
        ]

        # Prediction Heads (highest LR) - Consolidating all prediction layers
        prediction_head_params = [
            p
            for n, p in self.model.named_parameters()
            if any(
                head in n
                for head in [
                    "class_embed",
                    "enc_score_head",
                    "denoising_class_embed",
                    "bbox_embed",
                    "enc_bbox_head",
                ]
            )
            and p.requires_grad
        ]

        backbone_param_ids = {id(p) for p in backbone_params}
        encoder_decoder_param_ids = {id(p) for p in encoder_decoder_params}
        prediction_head_param_ids = {id(p) for p in prediction_head_params}

        # Remove overlaps
        encoder_decoder_params = [
            p
            for p in encoder_decoder_params
            if id(p) not in prediction_head_param_ids
            and id(p) not in backbone_param_ids
        ]
        prediction_head_params = [
            p
            for p in prediction_head_params
            if id(p) not in backbone_param_ids
            and id(p) not in encoder_decoder_param_ids
        ]
        backbone_params = [
            p
            for p in backbone_params
            if id(p) not in encoder_decoder_param_ids
            and id(p) not in prediction_head_param_ids
        ]

        # Everything else (Neck, FPN, PAN : medium LR)
        all_used_ids = (
            backbone_param_ids | encoder_decoder_param_ids | prediction_head_param_ids
        )
        other_params = [
            p
            for p in self.model.parameters()
            if id(p) not in all_used_ids and p.requires_grad
        ]

        param_dicts = []

        # Medium LR for other parameters
        if other_params:
            param_dicts.append(
                {"params": other_params, "lr": self.cfg.lr / self.cfg.lr_enc_dec_factor}
            )
            print(
                f"Neck and other params: {sum(p.numel() for p in other_params)} parameters at LR {self.cfg.lr}"
            )

        # Medium LR for encoder/decoder
        if encoder_decoder_params:
            param_dicts.append(
                {
                    "params": encoder_decoder_params,
                    "lr": self.cfg.lr / self.cfg.lr_enc_dec_factor,
                }
            )
            print(
                f"Encoder/Decoder params: {sum(p.numel() for p in encoder_decoder_params)} parameters at LR {self.cfg.lr / self.cfg.lr_enc_dec_factor}"
            )

        # Lowest LR for backbone
        if backbone_params:
            param_dicts.append(
                {"params": backbone_params, "lr": self.cfg.lr / self.cfg.lr_back_factor}
            )
            print(
                f"Backbone params: {sum(p.numel() for p in backbone_params)} parameters at LR {self.cfg.lr / self.cfg.lr_back_factor}"
            )

        # Highest LR for Prediction head (base LR)
        if prediction_head_params:
            param_dicts.append({"params": prediction_head_params, "lr": self.cfg.lr})
            print(
                f"Prediction head params: {sum(p.numel() for p in prediction_head_params)} parameters at LR {self.cfg.lr}"
            )

        optimizer = AdamW(param_dicts, weight_decay=self.cfg.weight_decay)
        return optimizer

    def move_labels_to_device(self, batch: BatchEncoding) -> BatchEncoding:
        """
        Moves label tensors within a batch to the specified device.

        Args:
            batch (BatchEncoding): The batch containing labels.

        Returns:
            BatchEncoding: The batch with labels moved to the device.
        """
        for lab in batch["labels"]:
            for k, v in lab.items():
                lab[k] = v.to(self.device)
        return batch

    def nested_to_cpu(self, obj: Any) -> Any:
        """
        Recursively moves tensors in a nested structure (dict, list, tuple) to CPU.

        Args:
            obj: The object containing tensors to move.

        Returns:
            The object with all tensors moved to CPU.
        """
        if torch.is_tensor(obj):
            return obj.cpu()
        if isinstance(obj, dict):
            return {k: self.nested_to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.nested_to_cpu(v) for v in obj]
        return obj

    def format_targets_for_cm(self, targets: List[Dict]) -> List[Dict]:
        """
        Formats raw targets for torchmetrics and confusion matrix.
        This is a helper for the confusion matrix, as MAPEvaluator handles its own formatting.
        """
        formatted_targets = []
        for target in targets:
            if "boxes" in target and "class_labels" in target:
                boxes = target["boxes"]
                labels = target["class_labels"]
                boxes = center_to_corners_format(boxes)
                width, height = target["size"][1], target["size"][0]
                boxes[:, [0, 2]] *= width
                boxes[:, [1, 3]] *= height
            else:
                boxes = torch.empty((0, 4))
                labels = torch.empty((0,))

            formatted_targets.append(
                {
                    "boxes": boxes.cpu(),
                    "labels": labels.cpu(),
                }
            )
        return formatted_targets

    def train(
        self,
        current_epoch: int,
    ) -> Dict[str, float]:
        """
        Performs one epoch of training.

        Args:
            current_epoch (int): The current epoch number.

        Returns:
            float: The average training loss for the epoch.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        epoch_loss = {"loss": 0.0}
        epoch_loss.update({k: 0.0 for k in ["class_loss", "bbox_loss", "giou_loss"]})

        progress_bar = tqdm(
            self.train_dl,
            desc=f"Epoch {current_epoch} Training",
            leave=False,
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
        )

        for batch_idx, batch in enumerate(progress_bar):
            batch["pixel_values"] = batch["pixel_values"].to(self.device)
            batch = self.move_labels_to_device(batch)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                out = self.model(**batch)
                batch_loss = out.loss / self.accum_steps
                loss_dict = out.loss_dict

            self.scaler.scale(batch_loss).backward()

            if (batch_idx + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            epoch_loss["loss"] += out.loss.item()
            epoch_loss["class_loss"] += loss_dict["loss_vfl"].item()
            epoch_loss["bbox_loss"] += loss_dict["loss_bbox"].item()
            epoch_loss["giou_loss"] += loss_dict["loss_giou"].item()

            current_avg_loss = epoch_loss["loss"] / (batch_idx + 1)
            progress_bar.set_postfix(
                {
                    "Loss": f"{current_avg_loss:.4f}",
                    "Batch": f"{batch_idx + 1}/{len(self.train_dl)}",
                }
            )

        num_batches = len(self.train_dl)
        epoch_loss = {k: v / num_batches for k, v in epoch_loss.items()}

        tqdm.write(f"Epoch : {current_epoch}")
        tqdm.write(
            f"\tTrain --- Loss: {epoch_loss['loss']:.4f}, Class Loss : {epoch_loss['class_loss']:.4f}, Bbox Loss : {epoch_loss['bbox_loss']:.4f}, Giou Loss : {epoch_loss['giou_loss']:.4f}"
        )

        return epoch_loss

    @torch.no_grad()
    def eval(
        self, test_dl: DataLoader, current_epoch: int, calc_cm: bool = False
    ) -> Tuple[dict[str, float], dict[str, Any], ConfusionMatrix | None]:
        """
        Evaluates the model on a given dataloader.

        Args:
            test_dl (DataLoader): The dataloader for evaluation.
            current_epoch (int): The current epoch number.
            calc_cm (bool, optional): Whether to calculate and return a confusion matrix. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - loss (Dict): The evaluation loss.
                - test_map (float): The mAP@.5-.95.
                - test_map50 (float): The mAP@.50.
                - test_map_50_per_class (torch.Tensor): The mAP@.50 for each class.
                - cm (ConfusionMatrix | None): The confusion matrix if calc_cm is True, else None.
        """
        self.model.eval()
        self.map_evaluator.map_metric.reset()
        self.map_evaluator.map_50_metric.reset()
        epoch_loss = {"loss": 0.0}
        epoch_loss.update({k: 0.0 for k in ["class_loss", "bbox_loss", "giou_loss"]})

        if calc_cm:
            cm = ConfusionMatrix(
                nc=len(test_dl.dataset.labels), conf=0.25, iou_thres=0.45
            )
        else:
            cm = None

        progress_bar = tqdm(
            test_dl,
            desc=f"Epoch {current_epoch} Evaluating",
            leave=False,
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
        )

        for batch_idx, batch in enumerate(progress_bar):
            batch["pixel_values"] = batch["pixel_values"].to(self.device)
            batch = self.move_labels_to_device(batch)

            out = self.model(**batch)
            batch_loss = out.loss
            loss_dict = out.loss_dict

            epoch_loss["loss"] += batch_loss.item()
            epoch_loss["class_loss"] += loss_dict["loss_vfl"].item()
            epoch_loss["bbox_loss"] += loss_dict["loss_bbox"].item()
            epoch_loss["giou_loss"] += loss_dict["loss_giou"].item()

            batch_targets = batch["labels"]
            image_sizes = self.map_evaluator.collect_image_sizes(batch_targets)

            batch_preds_processed = self.map_evaluator.collect_predictions(
                out, image_sizes
            )
            batch_targets_processed = self.map_evaluator.collect_targets(
                batch_targets, image_sizes
            )

            self.map_evaluator.map_metric.update(
                batch_preds_processed, batch_targets_processed
            )
            self.map_evaluator.map_50_metric.update(
                batch_preds_processed, batch_targets_processed
            )

            current_avg_loss = epoch_loss["loss"] / (batch_idx + 1)

            progress_bar.set_postfix(
                {
                    "Loss": f"{current_avg_loss:.4f}",
                    "Batch": f"{batch_idx + 1}/{len(test_dl)}",
                }
            )
            if calc_cm and cm:
                sizes = torch.stack(
                    [t["size"].clone().detach() for t in batch["labels"]]
                )
                preds = self.processor.post_process_object_detection(
                    out, threshold=0.01, target_sizes=sizes
                )
                preds = self.nested_to_cpu(preds)
                targets_for_cm = self.format_targets_for_cm(batch["labels"])
                cm.update(preds, targets_for_cm)

        tqdm.write("Computing mAP metrics")
        map_50_95_metrics = self.map_evaluator.map_metric.compute()
        test_map = map_50_95_metrics.get("map", 0.0)
        test_map50 = map_50_95_metrics.get("map_50", 0.0)
        map_50_metrics = self.map_evaluator.map_50_metric.compute()
        optimal_precisions, optimal_recalls = (
            self.map_evaluator.get_optimal_f1_ultralytics_style(map_50_metrics)
        )
        present_classes = map_50_metrics.get(
            "classes", torch.tensor([], device=self.device)
        )
        overall_precision, overall_recall = (
            self.map_evaluator.get_averaged_precision_recall_ultralytics_style(
                optimal_precisions, optimal_recalls, present_classes
            )
        )
        test_metrics = {
            "map@50:95": test_map,
            "map@50": test_map50,
            "map@50_per_class": self.map_evaluator.get_per_class(
                map_50_metrics, metric="map_per_class"
            ),
            "precision_per_class": optimal_precisions,
            "recall_per_class": optimal_recalls,
            "precision": overall_precision,
            "recall": overall_recall,
        }

        num_batches = len(test_dl)
        epoch_loss = {k: v / num_batches for k, v in epoch_loss.items()}

        tqdm.write(
            f"\tEval  --- Loss: {epoch_loss['loss']:.4f}, Class Loss : {epoch_loss['class_loss']:.4f}, Bbox Loss : {epoch_loss['bbox_loss']:.4f}, Giou Loss : {epoch_loss['giou_loss']:.4f}"
        )
        tqdm.write(f"\tEval  --- mAP50-95: {test_map:.4f}, mAP@50 : {test_map50:.4f}")

        tqdm.write("\t--- Per-class mAP@50 ---")
        class_names = test_dl.dataset.labels
        if test_metrics["map@50_per_class"].is_cuda:
            test_metrics["map@50_per_class"] = test_metrics["map@50_per_class"].cpu()

        for i, class_name in enumerate(class_names):
            if i < len(test_metrics["map@50_per_class"]):
                tqdm.write(
                    f"\t\t{class_name:<15}: {test_metrics['map@50_per_class'][i].item():.4f}"
                )

        return epoch_loss, test_metrics, cm

    def fit(self) -> None:
        """
        Runs the main training loop for the specified number of epochs.
        """
        epoch_pbar = tqdm(total=self.cfg.epochs, desc="Epochs", position=0, leave=True)

        best_test_map = 0

        for epoch in range(self.start_epoch, self.cfg.epochs):
            epoch_pbar.set_description(f"Epoch {epoch + 1}/{self.cfg.epochs}")
            if self.cfg.log and self.run and self.cfg.checkpoint:
                ckpt_path = self._save_checkpoint(epoch)
                log_checkpoint_artifact(
                    self.run, ckpt_path, self.cfg.model.name, epoch, self.cfg.wait
                )
            if isinstance(
                self.train_dl.dataset,
                AlbumentationsMosaicDataset,
            ):
                self.train_dl.dataset.update_epoch(epoch)

            train_loss = self.train(
                epoch + 1,
            )

            test_loss, test_metrics, _ = self.eval(self.test_dl, epoch + 1)

            self.scheduler.step()

            epoch_pbar.update(1)

            best_test_map = max(test_metrics["map@50"], best_test_map)

            if self.cfg.log:
                log_epoch_data(
                    epoch,
                    train_loss,
                    test_loss,
                    test_metrics,
                    self,
                )

            if (
                self.early_stopping(test_metrics["map@50:95"], self.model)
                and epoch > self.cfg.warmup_epochs + 5
            ):
                tqdm.write(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        tqdm.write("Training finished.")

        if self.cfg.log:
            log_val_data(epoch, best_test_map, self)
            self.run.finish()

        epoch_pbar.close()

    def _save_checkpoint(self, epoch: int) -> str:
        """
        Saves a checkpoint of the model, optimizer, scheduler, and scaler states.

        Args:
            epoch (int): The current epoch number.

        Returns:
            str: The path to the saved checkpoint file.
        """
        tqdm.write("saving checkpoint")
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        path = os.path.join("checkpoints", f"{self.name}_epoch{epoch}.pth")
        torch.save(ckpt, path)
        tqdm.write("done saving checkpoint")
        return path

    def _load_checkpoint(self, path: str, model_only: bool = True) -> None:
        """
        Loads a checkpoint and restores the state of the model, optimizer, scheduler, and scaler.

        Args:
            path (str): The path to the checkpoint file.
        """
        tqdm.write("loading checkpoint")
        ckpt = torch.load(path, map_location=self.device)
        if model_only:
            self.model.load_state_dict(ckpt)
        else:
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.scaler.load_state_dict(ckpt["scaler"])
            self.start_epoch = ckpt["epoch"] + 1
