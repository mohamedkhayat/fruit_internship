import torch
import torch.nn as nn
from torch.amp import GradScaler
import torchvision
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
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

    def to_coco_dict(self, idx: int, boxes, labels) -> dict:
        coco_ann = []
        for box, cls in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()
            coco_box = [x1, y1, x2 - x1, y2 - y1]
            area = (x2 - x1) * (y2 - y1)
            coco_ann.append(
                {
                    "category_id": int(cls),
                    "bbox": coco_box,
                    "area": area,
                    "iscrowd": 0,
                }
            )
        return {"image_id": idx, "annotations": coco_ann}

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

        for batch_idx, (x, y) in enumerate(progress_bar):
            coco_targets = [
                self.to_coco_dict(i, t["boxes"], t["labels"]) for i, t in enumerate(y)
            ]
            batch = self.processor(
                images=x,
                annotations=coco_targets,
                return_tensors="pt",
            ).to(self.device)

            batch = self.move_labels_to_device(batch)
            self.optimizer.zero_grad()

            with torch.autocast(device_type=device_str, dtype=torch.float16):
                out = self.model(**batch)
                batch_loss = out.loss

            self.scaler.scale(batch_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss += batch_loss.item()

            current_avg_loss = loss / (batch_idx + 1)
            progress_bar.set_postfix(
                {
                    "Loss": f"{current_avg_loss:.4f}",
                    "Batch": f"{batch_idx + 1}/{len(self.train_dl)}",
                }
            )

        loss /= len(self.train_dl)

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

        for batch_idx, (x, y) in enumerate(progress_bar):
            coco_targets = [
                self.to_coco_dict(i, t["boxes"], t["labels"]) for i, t in enumerate(y)
            ]
            batch = self.processor(
                images=x,
                annotations=coco_targets,
                return_tensors="pt",
            ).to(self.device)

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
            sizes = torch.tensor([[img.shape[0], img.shape[1]] for img in x]).to(
                self.device
            )
            preds = self.processor.post_process_object_detection(
                out, threshold=0.001, target_sizes=sizes
            )
            for p in preds:
                topk = p["scores"].argsort(descending=True)[:300]
                for k in ("boxes", "scores", "labels"):
                    p[k] = p[k][topk]

            preds = self.nested_to_cpu(preds)
            metric.update(preds, y)

        stats = metric.compute()
        metric.reset()

        loss /= len(test_dl)
        test_map, test_map50 = stats["map"].item(), stats["map_50"].item()

        tqdm.write(
            f"\tEval  --- Loss: {loss:.4f}, mAP50-95: {test_map:.4f}, mAP@50 : {test_map50:.4f}"
        )

        return loss, test_map, test_map50

    def fit(self):
        epoch_pbar = tqdm(total=self.cfg.epochs, desc="Epochs", position=0, leave=True)

        best_test_map = 0

        for epoch in range(self.cfg.epochs):
            epoch_pbar.set_description(f"Epoch {epoch + 1}/{self.cfg.epochs}")

            train_loss = self.train(
                epoch + 1,
            )

            test_loss, test_map, test_map50 = self.eval(self.test_dl, epoch + 1)

            self.scheduler.step()

            epoch_pbar.update(1)

            best_test_map = max(test_map50, best_test_map)

            if self.cfg.log:
                self.run.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "test/map": test_map,
                        "test/map 50": test_map50,
                        "test/loss": test_loss,
                        "Learning rate": float(
                            f"{self.scheduler.get_last_lr()[0]:.6f}"
                        ),
                    },
                )
            if self.early_stopping(test_map, self.model):
                tqdm.write(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        tqdm.write("Training finished.")

        if self.cfg.log:
            self.run.log({"test/best test map": best_test_map})
            # log_confusion_matrix(run, y_true, y_pred, labels)
            model = self.early_stopping.get_best_model(model)

            val_loss, val_map, val_map50 = self.eval(
                self.val_dl, epoch + 1
            )

            self.run.log(
                {"val/loss": val_loss, "val/map": val_map, "val/map@50": val_map50}
            )
            self.run.finish()

        tqdm.write(
            f"\tVal  --- Loss: {val_loss:.4f}, mAP50-95: {val_map:.4f}, mAP@50 : {val_map50:.4f}"
        )

        epoch_pbar.close()
