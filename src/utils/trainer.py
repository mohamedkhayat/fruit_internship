import os
import numpy as np
import torch
from torch.amp import GradScaler
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from utils.logging import log_checkpoint_artifact
from .early_stop import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR


class Trainer:
    def __init__(
        self, model, processor, device, cfg, name, run, train_dl, test_dl, val_dl
    ):
        self.model = model
        self.device = device
        self.scaler = GradScaler("cuda")
        self.cfg = cfg
        self.optimizer = self.get_optimizer()
        self.processor = processor
        self.name = name
        self.early_stopping = EarlyStopping(
            cfg.patience, cfg.delta, "checkpoints", name
        )
        self.scheduler = self.get_scheduler()
        self.run = run
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.val_dl = val_dl
        self.start_epoch = 0
        self.accum_steps = self.cfg.effective_batch_size // self.cfg.step_batch_size
        assert self.cfg.effective_batch_size % self.cfg.step_batch_size == 0, (
            f"effective_batch_size ({self.cfg.effective_batch_size}) must be divisible by batch_size "
            f"({self.step_batch_size})."
        )

    def get_scheduler(self):
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, total_iters=self.cfg.warmup_epochs
        )

        main_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.epochs - self.cfg.warmup_epochs, eta_min=1e-6
        )

        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.cfg.warmup_epochs],
        )
        return scheduler

    def get_optimizer(self):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.cfg.lr / self.cfg.lr_factor,
            },
        ]

        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        return optimizer

    def move_labels_to_device(self, batch):
        for lab in batch["labels"]:
            for k, v in lab.items():
                lab[k] = v.to(self.device)
        return batch

    def nested_to_cpu(self, obj):
        if torch.is_tensor(obj):
            return obj.cpu()
        if isinstance(obj, dict):
            return {k: self.nested_to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.nested_to_cpu(v) for v in obj]
        return obj

    def format_targets_for_map(self, y):
        y_metric_format = []
        for target_dict in y:
            annotations = target_dict["annotations"]

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
        current_epoch,
    ):
        self.model.train()
        loss = 0.0

        device_str = str(self.device).split(":")[0]
        progress_bar = tqdm(
            self.train_dl,
            desc=f"Epoch {current_epoch} Training",
            leave=False,
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
        )
        for batch_idx, (batch, _) in enumerate(progress_bar):
            batch = batch.to(self.device)
            batch = self.move_labels_to_device(batch)

            with torch.autocast(device_type=device_str, dtype=torch.float16):
                out = self.model(**batch)
                batch_loss = out.loss / self.accum_steps

            self.scaler.scale(batch_loss).backward()

            if (batch_idx + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            loss += batch_loss.item()

            current_avg_loss = loss / (batch_idx + 1)
            progress_bar.set_postfix(
                {
                    "Loss": f"{current_avg_loss:.4f}",
                    "Batch": f"{batch_idx + 1}/{len(self.train_dl)}",
                }
            )

        loss /= len(self.train_dl)

        tqdm.write(f"Epoch : {current_epoch}")
        tqdm.write(f"\tTrain --- Loss: {loss:.4f}")

        return loss

    @torch.no_grad()
    def eval(self, test_dl, current_epoch):
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

        device_str = str(self.device).split(":")[0]

        progress_bar = tqdm(
            test_dl,
            desc=f"Epoch {current_epoch} Evaluating",
            leave=False,
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
        )

        for batch_idx, (batch, y) in enumerate(progress_bar):
            batch = batch.to(self.device)
            batch = self.move_labels_to_device(batch)

            with torch.autocast(device_type=device_str, dtype=torch.float16):
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
            for p in preds:
                topk = p["scores"].argsort(descending=True)[:300]
                for k in ("boxes", "scores", "labels"):
                    p[k] = p[k][topk]

            preds = self.nested_to_cpu(preds)
            y = self.format_targets_for_map(y)
            metric.update(preds, y)

        stats = metric.compute()
        metric.reset()

        loss /= len(test_dl)
        test_map, test_map50, test_map_50_per_class = (
            stats["map"].item(),
            stats["map_50"].item(),
            stats["map_per_class"],
        )

        tqdm.write(
            f"\tEval  --- Loss: {loss:.4f}, mAP50-95: {test_map:.4f}, mAP@50 : {test_map50:.4f}"
        )

        tqdm.write("\t--- Per-class mAP@50 ---")
        class_names = test_dl.dataset.labels
        if test_map_50_per_class.is_cuda:
            test_map_50_per_class = test_map_50_per_class.cpu()

        for i, class_name in enumerate(class_names):
            if i < len(test_map_50_per_class):
                tqdm.write(
                    f"\t\t{class_name:<15}: {test_map_50_per_class[i].item():.4f}"
                )

        return loss, test_map, test_map50, test_map_50_per_class

    def fit(self):
        epoch_pbar = tqdm(total=self.cfg.epochs, desc="Epochs", position=0, leave=True)

        best_test_map = 0

        for epoch in range(self.start_epoch, self.cfg.epochs):
            epoch_pbar.set_description(f"Epoch {epoch + 1}/{self.cfg.epochs}")
            if self.cfg.log and self.run and self.cfg.checkpoint:
                ckpt_path = self._save_checkpoint(epoch)
                log_checkpoint_artifact(
                    self.run, ckpt_path, self.cfg.model.name, epoch, self.cfg.wait
                )

            train_loss = self.train(
                epoch + 1,
            )

            test_loss, test_map, test_map50, test_map_per_class = self.eval(
                self.test_dl, epoch + 1
            )

            self.scheduler.step()

            epoch_pbar.update(1)

            best_test_map = max(test_map50, best_test_map)

            if self.cfg.log:
                log_data = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "test/map": test_map,
                    "test/map 50": test_map50,
                    "test/loss": test_loss,
                    "Learning rate": float(f"{self.scheduler.get_last_lr()[0]:.6f}"),
                }

                class_names = self.test_dl.dataset.labels
                map_50_per_class = test_map_per_class.cpu()
                for i, name in enumerate(class_names):
                    if i < len(map_50_per_class):
                        log_data[f"test/map_50/{name}"] = map_50_per_class[i].item()

            self.run.log(log_data)

            if self.early_stopping(test_map, self.model):
                tqdm.write(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        tqdm.write("Training finished.")

        if self.cfg.log:
            self.model = self.early_stopping.get_best_model(self.model)

            val_loss, val_map, val_map50, val_map_50_per_class = self.eval(
                self.val_dl, epoch + 1
            )
            log_data = {
                "test/best test map": best_test_map,
                "val/loss": val_loss,
                "val/map": val_map,
                "val/map@50": val_map50,
            }

            tqdm.write("\t--- Per-class mAP@50 ---")
            class_names = self.val_dl.dataset.labels
            map_50_per_class = val_map_50_per_class.cpu()
            for i, name in enumerate(class_names):
                if i < len(map_50_per_class):
                    log_data[f"val/map_50/{name}"] = map_50_per_class[i].item()

            self.run.log(log_data)
            self.run.finish()

        tqdm.write(
            f"\tVal  --- Loss: {val_loss:.4f}, mAP50-95: {val_map:.4f}, mAP@50 : {val_map50:.4f}"
        )

        epoch_pbar.close()

    def _save_checkpoint(self, epoch):
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

    def _load_checkpoint(self, path):
        tqdm.write("loading checkpoint")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = ckpt["epoch"] + 1
