import torch
import torch.nn as nn
from torch.amp import GradScaler
import torchvision
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def get_optimizer(model, head_lr=1e-4, backbone_lr=1e-5, weight_decay=1e-4, **kwargs):
    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": backbone_lr,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=head_lr, weight_decay=weight_decay)
    return optimizer


def to_coco_dict(idx: int, boxes, labels) -> dict:
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


def move_labels_to_device(batch, device):
    for lab in batch["labels"]:
        for k, v in lab.items():
            lab[k] = v.to(device)
    return batch


def train(
    model: nn.Module,
    device: torch.device,
    train_dl,
    scaler: GradScaler,
    optimizer,
    current_epoch,
    processor,
):
    model.train()
    loss = 0.0

    device_str = str(device).split(":")[0]
    progress_bar = tqdm(
        train_dl,
        desc=f"Epoch {current_epoch} Training",
        leave=False,
        bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
    )

    for batch_idx, (x, y) in enumerate(progress_bar):
        coco_targets = [
            to_coco_dict(i, t["boxes"], t["labels"]) for i, t in enumerate(y)
        ]
        batch = processor(
            images=x,
            annotations=coco_targets,
            return_tensors="pt",
        ).to(device)
        batch = move_labels_to_device(batch, device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device_str, dtype=torch.float16):
            out = model(**batch)
            batch_loss = out.loss

        scaler.scale(batch_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss += batch_loss.item()

        current_avg_loss = loss / (batch_idx + 1)
        progress_bar.set_postfix(
            {
                "Loss": f"{current_avg_loss:.4f}",
                "Batch": f"{batch_idx + 1}/{len(train_dl)}",
            }
        )

    loss /= len(train_dl)
    return loss


@torch.no_grad()
def eval(model: nn.Module, device, test_dl, current_epoch, processor):
    model.eval()
    metric = MeanAveragePrecision(
        box_format="xyxy",
        average="macro",
        max_detection_thresholds=[1, 10, 100],
        iou_thresholds=None,
        class_metrics=True,
    )
    metric.warn_on_many_detections = False
    loss = 0.0

    device_str = str(device).split(":")[0]

    progress_bar = tqdm(
        test_dl,
        desc=f"Epoch {current_epoch} Evaluating",
        leave=False,
        bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
    )

    for batch_idx, (x, y) in enumerate(progress_bar):
        coco_targets = [
            to_coco_dict(i, t["boxes"], t["labels"]) for i, t in enumerate(y)
        ]
        batch = processor(
            images=x,
            annotations=coco_targets,
            return_tensors="pt",
        ).to(device)

        batch = move_labels_to_device(batch, device)

        with torch.autocast(device_type=device_str, dtype=torch.float16):
            out = model(**batch)
            batch_loss = out.loss

        loss += batch_loss.item()

        current_avg_loss = loss / (batch_idx + 1)

        progress_bar.set_postfix(
            {
                "Loss": f"{current_avg_loss:.4f}",
                "Batch": f"{batch_idx + 1}/{len(test_dl)}",
            }
        )
        sizes = torch.tensor([[img.shape[0], img.shape[1]] for img in x]).to(device)
        preds = processor.post_process_object_detection(
            out, threshold=0.001, target_sizes=sizes
        )
        for p in preds:
            topk = p["scores"].argsort(descending=True)[:300]
            for k in ("boxes", "scores", "labels"):
                p[k] = p[k][topk]

        preds = nested_to_cpu(preds)
        metric.update(preds, y)

    stats = metric.compute()
    metric.reset()

    loss /= len(test_dl)
    return loss, stats["map"].item(), stats["map_50"].item()


def nested_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: nested_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [nested_to_cpu(v) for v in obj]
    return obj
