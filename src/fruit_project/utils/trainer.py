# SPDX-FileCopyrightText: 2025 Mohamed Khayat
# SPDX-License-Identifier: AGPL-3.0-or-later

import gc
import os
from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
from torch.amp import GradScaler
from tqdm import tqdm
from transformers.image_transforms import center_to_corners_format
from omegaconf import DictConfig
from fruit_project.utils.datasets.alb_mosaic_dataset import AlbumentationsMosaicDataset
from fruit_project.utils.early_stop import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from fruit_project.utils.metrics import ConfusionMatrix, MAPEvaluator
from transformers import AutoImageProcessor, BatchEncoding
from wandb.sdk.wandb_run import Run
from torch.utils.data import DataLoader
from torch.optim import AdamW, Optimizer
from torch_ema import ExponentialMovingAverage
from fruit_project.utils.logging import (
    log_checkpoint_artifact,
    log_epoch_data,
    log_test_data,
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
        loading_info: Dict,
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
        self.optimizer: Optimizer = self.get_optimizer(loading_info)
        self.processor: AutoImageProcessor = processor
        self.name: str = name
        self.early_stopping: EarlyStopping = EarlyStopping(
            cfg.patience, cfg.delta, "checkpoints", name, run, cfg.log, cfg.upload
        )
        self.run: Run = run
        self.train_dl: DataLoader = train_dl
        self.test_dl: DataLoader = test_dl
        self.val_dl: DataLoader = val_dl
        self.start_epoch: int = 0
        self.map_evaluator = MAPEvaluator(
            image_processor=processor,
            device=self.device,
            threshold=self.cfg.threshold,
            id2label=train_dl.dataset.id2lbl,
        )
        self.accum_steps: int = (
            self.cfg.effective_batch_size // self.cfg.step_batch_size
        )
        assert self.cfg.effective_batch_size % self.cfg.step_batch_size == 0, (
            f"effective_batch_size ({self.cfg.effective_batch_size}) must be divisible by batch_size "
            f"({self.accum_steps})."
        )
        self.scheduler: SequentialLR = self.get_scheduler()
        self.ema: ExponentialMovingAverage = (
            None
            if not self.cfg.ema.use
            else ExponentialMovingAverage(
                self.model.parameters(), decay=self.cfg.ema.decay
            )
        )

    def get_scheduler(self) -> SequentialLR:
        """
        Creates a learning rate scheduler with a warmup phase.

        Returns:
            SequentialLR: The learning rate scheduler.
        """
        train_steps = len(self.train_dl) // self.accum_steps

        if self.cfg.phase == 1:
            total_warmup_steps = self.cfg.warmup_epochs * train_steps
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=self.cfg.lin_start_factor,
                total_iters=total_warmup_steps,
            )

            main_steps = (self.cfg.epochs - self.cfg.warmup_epochs) * train_steps
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=main_steps,
                eta_min=self.cfg.lr / self.cfg.eta_min_factor,
            )

            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[
                    total_warmup_steps,
                ],
            )
        else:
            scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.epochs * train_steps, eta_min=0
            )
        return scheduler

    def get_optimizer(self, loading_info=None) -> AdamW:
        """
        Creates an AdamW optimizer with a differential learning rate for the backbone
        and the rest of the model (head), following standard fine-tuning practices.

        Returns:
            AdamW: The configured optimizer.
        """
        if not self.cfg.smart_optim:
            head_params = [p for p in self.model.parameters() if p.requires_grad]

            backbone_params = [
                p
                for n, p in self.model.named_parameters()
                if n.startswith("model.backbone")
                or n.startswith("vit")
                and p.requires_grad
            ]

            backbone_param_ids = {id(p) for p in backbone_params}

            head_params_final = [
                p for p in head_params if id(p) not in backbone_param_ids
            ]

            param_dicts = [
                {
                    "params": head_params_final,
                    "lr": self.cfg.lr,
                },
                {
                    "params": backbone_params,
                    "lr": self.cfg.lr / self.cfg.lr_back_factor,
                },
            ]

        else:
            mismatched_keys = set(loading_info.get("mismatched_keys", []))
            missing_keys = set(loading_info.get("missing_keys", []))

            new_param_names = mismatched_keys.union(missing_keys)

            head_params_final = []
            backbone_params = []

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                param_is_new = any(key in name for key in new_param_names)

                if param_is_new:
                    head_params_final.append(param)
                else:
                    backbone_params.append(param)

        param_dicts = [
            {
                "params": head_params_final,
                "lr": self.cfg.lr,
            },
            {
                "params": backbone_params,
                "lr": self.cfg.lr / self.cfg.lr_back_factor,
            },
        ]
        print(
            f"Backbone params: {sum(p.numel() for p in backbone_params)} parameters at LR {self.cfg.lr / self.cfg.lr_back_factor}"
        )
        print(
            f"Head (Encoder, Decoder, Neck, etc.) params: {sum(p.numel() for p in head_params_final)} parameters at LR {self.cfg.lr}"
        )

        if self.cfg.optim == "8bit":
            tqdm.write("using AdamW8bit")
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                param_dicts,
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optim == "torch":
            tqdm.write("using torch AdamW")
            optimizer = AdamW(
                param_dicts,
                weight_decay=self.cfg.weight_decay,
                fused=True,
            )
        else:
            raise KeyError("invalid optim type, use 8bit or torch")

        return optimizer

    def move_batch_to_device(self, batch: BatchEncoding) -> BatchEncoding:
        """
        Moves label tensors within a batch to the specified device.

        Args:
            batch (BatchEncoding): The batch containing labels.

        Returns:
            BatchEncoding: The batch with labels moved to the device.
        """

        batch = {k: v.to(self.device) if k != "labels" else v for k, v in batch.items()}

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
        epoch_loss.update(
            {
                k: 0.0
                for k in ["class_loss", "bbox_loss", "giou_loss", "cardinality_error"]
            }
        )

        progress_bar = tqdm(
            self.train_dl,
            desc=f"Epoch {current_epoch} Training",
            leave=False,
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
        )

        for batch_idx, batch in enumerate(progress_bar):
            batch = self.move_batch_to_device(batch)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                out = self.model(**batch)
                batch_loss = out.loss / self.accum_steps
                loss_dict = out.loss_dict

            self.scaler.scale(batch_loss).backward()

            if (batch_idx + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.cfg.model.grad_max_norm
                )
                self.scaler.step(self.optimizer)
                self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.ema and current_epoch >= self.cfg.warmup_epochs:
                    self.ema.update()

            epoch_loss["loss"] += out.loss.item()
            epoch_loss["class_loss"] += loss_dict.get(
                "loss_vfl", loss_dict.get("loss_ce", torch.tensor(0.0))
            ).item()
            epoch_loss["bbox_loss"] += loss_dict["loss_bbox"].item()
            epoch_loss["giou_loss"] += loss_dict["loss_giou"].item()
            epoch_loss["cardinality_error"] += loss_dict.get(
                "cardinality_error", torch.tensor(0.0)
            ).item()

            current_avg_loss = epoch_loss["loss"] / (batch_idx + 1)
            progress_bar.set_postfix(
                {
                    "Loss": f"{current_avg_loss:.4f}",
                    "Batch": f"{batch_idx + 1}/{len(self.train_dl)}",
                }
            )

            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        num_batches = len(self.train_dl)
        epoch_loss = {k: v / num_batches for k, v in epoch_loss.items()}

        tqdm.write(f"Epoch : {current_epoch}")
        tqdm.write(
            f"\tTrain --- Loss: {epoch_loss['loss']:.4f}, Class Loss : {epoch_loss['class_loss']:.4f}, Bbox Loss : {epoch_loss['bbox_loss']:.4f}, Giou Loss : {epoch_loss['giou_loss']:.4f}"
        )

        return epoch_loss

    @torch.no_grad()
    def eval(
        self, val_dl: DataLoader, current_epoch: int, calc_cm: bool = False
    ) -> Tuple[dict[str, float], dict[str, Any], Optional[ConfusionMatrix]]:
        if self.ema and current_epoch >= self.cfg.warmup_epochs:
            tqdm.write("evaluating with EMA weights")
            with self.ema.average_parameters():
                return self._run_eval(val_dl, current_epoch, calc_cm)
        else:
            tqdm.write("evaluating with regular weights")
            return self._run_eval(val_dl, current_epoch, calc_cm)

    @torch.no_grad()
    def _run_eval(
        self, val_dl: DataLoader, current_epoch: int, calc_cm: bool = False
    ) -> Tuple[dict[str, float], dict[str, Any], Optional[ConfusionMatrix]]:
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
        epoch_loss.update(
            {
                k: 0.0
                for k in ["class_loss", "bbox_loss", "giou_loss", "cardinality_error"]
            }
        )

        if calc_cm:
            cm = ConfusionMatrix(
                nc=len(val_dl.dataset.labels), conf=0.374, iou_thres=0.45
            )
        else:
            cm = None

        progress_bar = tqdm(
            val_dl,
            desc=f"Epoch {current_epoch} Evaluating",
            leave=False,
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
        )

        for batch_idx, batch in enumerate(progress_bar):
            batch = self.move_batch_to_device(batch)

            out = self.model(**batch)
            batch_loss = out.loss
            loss_dict = out.loss_dict

            epoch_loss["loss"] += batch_loss.item()
            epoch_loss["class_loss"] += loss_dict.get(
                "loss_vfl", loss_dict.get("loss_ce", torch.tensor(0.0))
            ).item()
            epoch_loss["bbox_loss"] += loss_dict["loss_bbox"].item()
            epoch_loss["giou_loss"] += loss_dict["loss_giou"].item()
            epoch_loss["cardinality_error"] += loss_dict.get(
                "cardinality_error", torch.tensor(0.0)
            ).item()

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
                    "Batch": f"{batch_idx + 1}/{len(val_dl)}",
                }
            )

            if calc_cm and cm:
                preds = self.nested_to_cpu(batch_preds_processed)
                targets_for_cm = self.format_targets_for_cm(batch["labels"])
                cm.update(preds, targets_for_cm)

            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

            del batch_preds_processed, batch_targets_processed

        tqdm.write("Computing mAP metrics")
        map_50_95_metrics = self.map_evaluator.map_metric.compute()
        val_map = map_50_95_metrics.get("map", 0.0)
        val_map50 = map_50_95_metrics.get("map_50", 0.0)
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
        val_metrics = {
            "map@50:95": val_map,
            "map@50": val_map50,
            "map@50_per_class": self.map_evaluator.get_per_class(
                map_50_metrics, metric="map_per_class"
            ),
            "precision_per_class": optimal_precisions,
            "recall_per_class": optimal_recalls,
            "precision": overall_precision,
            "recall": overall_recall,
        }

        num_batches = len(val_dl)
        epoch_loss = {k: v / num_batches for k, v in epoch_loss.items()}

        tqdm.write(
            f"\tEval  --- Loss: {epoch_loss['loss']:.4f}, Class Loss : {epoch_loss['class_loss']:.4f}, Bbox Loss : {epoch_loss['bbox_loss']:.4f}, Giou Loss : {epoch_loss['giou_loss']:.4f}"
        )
        tqdm.write(f"\tEval  --- mAP50-95: {val_map:.4f}, mAP@50 : {val_map50:.4f}")

        tqdm.write("\t--- Per-class mAP@50 ---")
        class_names = val_dl.dataset.labels
        if val_metrics["map@50_per_class"].is_cuda:
            val_metrics["map@50_per_class"] = val_metrics["map@50_per_class"].cpu()

        for i, class_name in enumerate(class_names):
            if i < len(val_metrics["map@50_per_class"]):
                tqdm.write(
                    f"\t\t{class_name:<15}: {val_metrics['map@50_per_class'][i].item():.4f}"
                )

        del map_50_95_metrics, map_50_metrics
        del optimal_precisions, optimal_recalls, present_classes

        return epoch_loss, val_metrics, cm

    def fit(self) -> None:
        """
        Runs the main training loop for the specified number of epochs.
        """
        epoch_pbar = tqdm(total=self.cfg.epochs, desc="Epochs", position=0, leave=True)

        best_val_map = 0

        for epoch in range(self.start_epoch, self.cfg.epochs):
            epoch_pbar.set_description(f"Epoch {epoch}/{self.cfg.epochs}")
            if self.run and self.cfg.ckpt.save:
                ckpt_path = self._save_checkpoint(epoch)
                if self.cfg.log:
                    log_checkpoint_artifact(
                        self.run, ckpt_path, self.cfg.model.name, epoch, self.cfg.wait
                    )

            if isinstance(self.train_dl.dataset, AlbumentationsMosaicDataset):
                self.train_dl.dataset.update_epoch(epoch)

            train_loss = self.train(
                epoch,
            )

            val_loss, val_metrics, _ = self.eval(self.val_dl, epoch)

            epoch_pbar.update(1)

            if epoch % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            best_val_map = max(val_metrics["map@50"], best_val_map)

            if self.cfg.log:
                log_epoch_data(
                    epoch,
                    train_loss,
                    val_loss,
                    val_metrics,
                    self,
                )

            if self.early_stopping(val_metrics["map@50:95"], self.model):
                tqdm.write(f"Early stopping triggered at epoch {epoch}.")
                break

        tqdm.write("Training finished.")

        if self.cfg.log:
            log_test_data(epoch, best_val_map, self)
            self.run.finish()

        epoch_pbar.close()

    def _build_save_dict(self, epoch):
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "ema": self.ema.state_dict(),
        }
        return ckpt

    def _save_checkpoint(self, epoch: int) -> str:
        """
        Saves a checkpoint of the model, optimizer, scheduler, and scaler states.

        Args:
            epoch (int): The current epoch number.

        Returns:
            str: The path to the saved checkpoint file.
        """
        tqdm.write("saving checkpoint")
        if self.ema:
            with self.ema.average_parameters():
                ckpt = self._build_save_dict(epoch)
            ckpt.update({"ema": self.ema.state_dict()})
        else:
            ckpt = self._build_save_dict(epoch)

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
            if "ema" in ckpt:
                self.ema.load_state_dict(ckpt["ema"])
            self.start_epoch = ckpt["epoch"] + 1
