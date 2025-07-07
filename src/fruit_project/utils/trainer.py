import os
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.amp import GradScaler
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from omegaconf import DictConfig
from fruit_project.utils.logging import (
    log_checkpoint_artifact,
    log_detection_confusion_matrix,
)
from fruit_project.utils.early_stop import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from fruit_project.utils.metrics import ConfusionMatrix
from transformers import AutoImageProcessor, BatchEncoding
import torch.nn as nn
from wandb.sdk.wandb_run import Run
from torch.utils.data import DataLoader
from torch.optim import AdamW


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
        self.optimizer: torch.optim = self.get_optimizer()
        self.processor: AutoImageProcessor = processor
        self.name: str = name
        self.early_stopping: EarlyStopping = EarlyStopping(
            cfg.patience, cfg.delta, "checkpoints", name, cfg, run
        )
        self.scheduler: SequentialLR = self.get_scheduler()
        self.run: Run = run
        self.train_dl: DataLoader = train_dl
        self.test_dl: DataLoader = test_dl
        self.val_dl: DataLoader = val_dl
        self.start_epoch: int = 0
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
        non_backbone_params = [
            p
            for n, p in self.model.named_parameters()
            if "backbone" not in n and p.requires_grad
        ]

        backbone_params = [
            p
            for n, p in self.model.named_parameters()
            if "backbone" in n and p.requires_grad
        ]

        param_dicts = []

        if non_backbone_params:
            param_dicts.append({"params": non_backbone_params})

        if backbone_params:
            param_dicts.append(
                {"params": backbone_params, "lr": self.cfg.lr / self.cfg.lr_factor}
            )

        optimizer = AdamW(
            param_dicts, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
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

    def format_targets_for_map(self, y: List) -> List:
        """
        Formats target annotations for MeanAveragePrecision metric calculation.

        Args:
            y (List): A list of target dictionaries.

        Returns:
            List: A list of formatted target dictionaries for the metric.
        """
        y_metric_format = []
        for target_dict in y:
            annotations = target_dict["annotations"]

            if not annotations:
                y_metric_format.append(
                    {
                        "boxes": torch.empty((0, 4), dtype=torch.float32),
                        "labels": torch.empty((0,), dtype=torch.int64),
                    }
                )
                continue

            boxes_xywh = np.array(
                [ann["bbox"] for ann in annotations], dtype=np.float32
            )
            labels = np.array(
                [ann["category_id"] for ann in annotations], dtype=np.int64
            )

            # Convert from [x, y, w, h] to [x1, y1, x2, y2]
            boxes_xyxy = boxes_xywh.copy()
            boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]  # x2 = x1 + w
            boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]  # y2 = y1 + h
            boxes = boxes_xyxy

            y_metric_format.append(
                {
                    "boxes": torch.from_numpy(boxes),
                    "labels": torch.from_numpy(labels),
                }
            )
        return y_metric_format

    def train(
        self,
        current_epoch: int,
    ) -> Tuple[float, float, float, torch.Tensor]:
        """
        Performs one epoch of training.

        Args:
            current_epoch (int): The current epoch number.

        Returns:
            float: The average training loss for the epoch.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        metric = MeanAveragePrecision(
            box_format="xyxy",
            average="macro",
            max_detection_thresholds=[1, 10, 100],
            iou_thresholds=None,
            class_metrics=True,
        )
        metric.warn_on_many_detections = False
        loss = 0.0

        progress_bar = tqdm(
            self.train_dl,
            desc=f"Epoch {current_epoch} Training",
            leave=False,
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
        )

        for batch_idx, (batch, y) in enumerate(progress_bar):
            batch = batch.to(self.device)
            batch = self.move_labels_to_device(batch)

            with torch.autocast(
                device_type=self.device.device_type, dtype=torch.bfloat16
            ):
                out = self.model(**batch)
                batch_loss = out.loss / self.accum_steps

            self.scaler.scale(batch_loss).backward()

            if (batch_idx + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            loss += out.loss.item()

            current_avg_loss = loss / (batch_idx + 1)
            progress_bar.set_postfix(
                {
                    "Loss": f"{current_avg_loss:.4f}",
                    "Batch": f"{batch_idx + 1}/{len(self.train_dl)}",
                }
            )
            sizes = torch.stack([t["orig_size"] for t in y])
            preds = self.processor.post_process_object_detection(
                out, threshold=0.001, target_sizes=sizes
            )

            preds = self.nested_to_cpu(preds)
            targets_for_map = self.format_targets_for_map(y)
            preds_for_map = [p.copy() for p in preds]
            for p in preds_for_map:
                topk = p["scores"].argsort(descending=True)[:300]
                for k in ("boxes", "scores", "labels"):
                    p[k] = p[k][topk]

            metric.update(preds_for_map, targets_for_map)

        stats = metric.compute()
        metric.reset()

        loss /= len(self.train_dl)
        train_map, train_map50, train_map_per_class = (
            stats["map"].item(),
            stats["map_50"].item(),
            stats["map_per_class"],
        )

        tqdm.write(f"Epoch : {current_epoch}")
        tqdm.write(
            f"\tTrain --- Loss: {loss:.4f}, mAP50-95: {train_map:.4f}, mAP@50 : {train_map50:.4f}"
        )

        tqdm.write("\t--- Per-class mAP@50-95 ---")
        class_names = self.train_dl.dataset.labels
        if train_map_per_class.is_cuda:
            train_map_per_class = train_map_per_class.cpu()

        for i, class_name in enumerate(class_names):
            if i < len(train_map_per_class):
                tqdm.write(f"\t\t{class_name:<15}: {train_map_per_class[i].item():.4f}")

        return loss, train_map, train_map50, train_map_per_class

    @torch.no_grad()
    def eval(
        self, test_dl: DataLoader, current_epoch: int, calc_cm: bool = False
    ) -> Tuple[float, float, float, torch.Tensor, Optional[ConfusionMatrix]]:
        """
        Evaluates the model on a given dataloader.

        Args:
            test_dl (DataLoader): The dataloader for evaluation.
            current_epoch (int): The current epoch number.
            calc_cm (bool, optional): Whether to calculate and return a confusion matrix. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - loss (float): The average evaluation loss.
                - test_map (float): The mAP@.5-.95.
                - test_map50 (float): The mAP@.50.
                - test_map_50_per_class (torch.Tensor): The mAP@.50 for each class.
                - cm (ConfusionMatrix | None): The confusion matrix if calc_cm is True, else None.
        """
        self.model.eval()
        metric = MeanAveragePrecision(
            box_format="xyxy",
            average="macro",
            max_detection_thresholds=[1, 10, 100],
            iou_thresholds=None,
            class_metrics=True,
        )
        metric.warn_on_many_detections = False
        loss = 0.0
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

        for batch_idx, (batch, y) in enumerate(progress_bar):
            batch = batch.to(self.device)
            batch = self.move_labels_to_device(batch)

            out = self.model(**batch)
            batch_loss = out.loss

            loss += batch_loss.item()

            current_avg_loss = loss / (batch_idx + 1)

            progress_bar.set_postfix(
                {
                    "Loss": f"{current_avg_loss:.4f}",
                    "Batch": f"{batch_idx + 1}/{len(test_dl)}",
                }
            )
            sizes = torch.stack([t["orig_size"] for t in y])
            preds = self.processor.post_process_object_detection(
                out, threshold=0.001, target_sizes=sizes
            )

            preds = self.nested_to_cpu(preds)
            targets_for_map = self.format_targets_for_map(y)
            preds_for_map = [p.copy() for p in preds]
            for p in preds_for_map:
                topk = p["scores"].argsort(descending=True)[:300]
                for k in ("boxes", "scores", "labels"):
                    p[k] = p[k][topk]

            metric.update(preds_for_map, targets_for_map)

            if calc_cm and cm:
                for i in range(len(preds)):
                    pred_item = preds[i]
                    gt_item = targets_for_map[i]

                    detections = torch.cat(
                        [
                            pred_item["boxes"],
                            pred_item["scores"].unsqueeze(1),
                            pred_item["labels"].unsqueeze(1).float(),
                        ],
                        dim=1,
                    )

                    gt_boxes = gt_item["boxes"]
                    gt_labels = gt_item["labels"]

                    if gt_boxes.numel() > 0:
                        labels = torch.cat(
                            [gt_labels.unsqueeze(1).float(), gt_boxes], dim=1
                        )
                    else:
                        labels = torch.zeros((0, 5))

                    cm.process_batch(detections, labels)

        stats = metric.compute()
        metric.reset()

        loss /= len(test_dl)
        test_map, test_map50, test_map_per_class = (
            stats["map"].item(),
            stats["map_50"].item(),
            stats["map_per_class"],
        )

        tqdm.write(
            f"\tEval  --- Loss: {loss:.4f}, mAP50-95: {test_map:.4f}, mAP@50 : {test_map50:.4f}"
        )

        tqdm.write("\t--- Per-class mAP@50-95 ---")
        class_names = test_dl.dataset.labels
        if test_map_per_class.is_cuda:
            test_map_per_class = test_map_per_class.cpu()

        for i, class_name in enumerate(class_names):
            if i < len(test_map_per_class):
                tqdm.write(f"\t\t{class_name:<15}: {test_map_per_class[i].item():.4f}")

        return loss, test_map, test_map50, test_map_per_class, cm

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

            train_loss, train_map, train_map50, train_map_per_class = self.train(
                epoch + 1,
            )

            test_loss, test_map, test_map50, test_map_per_class, _ = self.eval(
                self.test_dl, epoch + 1
            )

            self.scheduler.step()

            epoch_pbar.update(1)

            best_test_map = max(test_map50, best_test_map)

            if self.cfg.log:
                log_data = self.get_epoch_log_data(
                    epoch,
                    train_loss,
                    train_map,
                    train_map50,
                    train_map_per_class,
                    test_map,
                    test_map50,
                    test_loss,
                    test_map_per_class,
                )
                self.run.log(log_data)

            if self.early_stopping(test_map, self.model):
                tqdm.write(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        tqdm.write("Training finished.")

        if self.cfg.log:
            log_data = self.get_val_log_data(epoch, best_test_map)
            self.run.log(log_data)
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

    def _load_checkpoint(self, path: str) -> None:
        """
        Loads a checkpoint and restores the state of the model, optimizer, scheduler, and scaler.

        Args:
            path (str): The path to the checkpoint file.
        """
        tqdm.write("loading checkpoint")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = ckpt["epoch"] + 1

    def get_epoch_log_data(
        self,
        epoch: int,
        train_loss: float,
        train_map: float,
        train_map50: float,
        train_map_per_class: torch.Tensor,
        test_map: float,
        test_map50: float,
        test_loss: float,
        test_map_per_class: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Constructs a dictionary of metrics for logging at the end of an epoch.

        Args:
            epoch (int): The current epoch number.
            train_loss (float): The training loss.
            test_map (float): The test mAP@.5-.95.
            test_map50 (float): The test mAP@.50.
            test_loss (float): The test loss.
            test_map_per_class (torch.Tensor): The test mAP@.50 for each class.

        Returns:
            dict: A dictionary of metrics for logging.
        """
        log_data = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/map": train_map,
            "train/map 50": train_map50,
            "test/map": test_map,
            "test/map 50": test_map50,
            "test/loss": test_loss,
            "Learning rate": float(f"{self.scheduler.get_last_lr()[0]:.6f}"),
        }

        self.log_per_class_map(
            self.test_dl.dataset.labels, test_map_per_class, "test", log_data
        )
        self.log_per_class_map(
            self.train_dl.dataset.labels, train_map_per_class, "train", log_data
        )

        return log_data

    def get_val_log_data(self, epoch: int, best_test_map: float) -> Dict[str, Any]:
        """
        Performs final validation, logs metrics, and returns the log data.

        Args:
            epoch (int): The final epoch number.
            best_test_map (float): The best test mAP@.50 achieved during training.

        Returns:
            dict: A dictionary of validation metrics for logging.
        """
        self.model = self.early_stopping.get_best_model(self.model)

        val_loss, val_map, val_map50, val_map_per_class, cm = self.eval(
            self.val_dl, epoch + 1, calc_cm=True
        )
        log_data = {
            "test/best test map": best_test_map,
            "val/loss": val_loss,
            "val/map": val_map,
            "val/map@50": val_map50,
        }

        tqdm.write("\t--- Per-class mAP@50-95 ---")
        class_names = self.val_dl.dataset.labels
        map_per_class = val_map_per_class.cpu()
        for i, name in enumerate(class_names):
            if i < len(map_per_class):
                log_data[f"val/map_50-95/{name}"] = map_per_class[i].item()

        tqdm.write(
            f"\tVal  --- Loss: {val_loss:.4f}, mAP50-95: {val_map:.4f}, mAP@50 : {val_map50:.4f}"
        )
        log_detection_confusion_matrix(self.run, cm, list(self.val_dl.dataset.labels))
        return log_data

    def log_per_class_map(self, class_names : List, map_per_class : torch.Tensor, ds_type : str, log_data : Dict) -> None:
        map_per_class = map_per_class.cpu()
        for i, name in enumerate(class_names):
            if i < len(map_per_class):
                log_data[f"{ds_type}/map_50-95/{name}"] = map_per_class[i].item()
