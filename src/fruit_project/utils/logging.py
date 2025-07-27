# SPDX-FileCopyrightText: 2025 Mohamed Khayat
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections import Counter
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from omegaconf import DictConfig
from wandb.sdk.wandb_run import Run
from typing import Dict, Tuple, List, Optional
from fruit_project.utils.metrics import ConfusionMatrix
from transformers.image_transforms import center_to_corners_format
import datetime

def initwandb(cfg: DictConfig) -> Run:
    """
    Initializes a wandb run.

    Args:
        cfg (DictConfig): Configuration object.

    Returns:
        Run: The wandb run object.
    """
    name = get_run_name(cfg)
    run = wandb.init(
        entity="mohamedkhayat025-none",
        project="fruit-transformer",
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    if cfg.info:
        info = str(input("any additional info ?"))
        run.summary["INFO"] = info
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("test/*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")

    return run


def get_run_name(cfg: DictConfig) -> str:
    """
    Generates a run name based on the configuration.

    Args:
        cfg (DictConfig): Configuration object.

    Returns:
        str: The generated run name.
    """
    now = datetime.datetime.now()
    date_str = now.strftime("%m%d_%H%M")
    name = f"model={cfg.model.name}_lr={cfg.lr}_{date_str}"
    return name


def log_images(
    run: Run,
    batch: Dict,
    id2lbl: Dict,
    grid_size: Tuple[int, int] = (3, 3),
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> None:
    """
    Logs a grid of images with their bounding boxes to wandb.

    Args:
        run (Run): The wandb run object.
        batch (Tuple[Dict, List]): A single batch of data (processed_batch, targets).
        id2lbl (Dict): A dictionary mapping class IDs to labels.
        grid_size (Tuple[int, int], optional): The grid size for displaying images. Defaults to (3, 3).
        mean (Optional[torch.Tensor], optional): The mean used for normalization. Defaults to None.
        std (Optional[torch.Tensor], optional): The standard deviation used for normalization. Defaults to None.
    """
    images = batch["pixel_values"].detach().clone()
    targets = batch["labels"]
    n_rows, n_cols = grid_size
    max_plots = n_rows * n_cols
    n = min(len(images), max_plots)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
    axes = axes.flatten()

    for ax in axes[n:]:
        ax.axis("off")

    for i in range(n):
        img = images[i]
        # img = unnormalize(img, mean, std).squeeze(0)
        tgt = targets[i]

        img_uint8 = (img * 255).to(torch.uint8)

        if "boxes" in tgt and "class_labels" in tgt:
            boxes_cywh = tgt["boxes"]

            if boxes_cywh.numel() == 0:
                boxes_to_draw = torch.empty((0, 4), dtype=torch.long)
            else:
                boxes_xyxy = center_to_corners_format(boxes_cywh)
                if boxes_xyxy.max() <= 1.0:
                    h, w = img_uint8.shape[1:]
                    boxes_xyxy[:, [0, 2]] *= w
                    boxes_xyxy[:, [1, 3]] *= h
                boxes_to_draw = boxes_xyxy.long()

            labels = [str(id2lbl[int(lbl)]) for lbl in tgt["class_labels"]]
        else:
            boxes_to_draw = torch.empty((0, 4), dtype=torch.long)
            labels = []

        annotated = draw_bounding_boxes(
            img_uint8,
            boxes=boxes_to_draw,
            labels=np.array(labels),
            colors="red",
            width=2,
            font="fonts/FiraCodeNerdFont-Bold.ttf",
            font_size=30,
        )

        axes[i].imshow(to_pil_image(annotated))
        axes[i].axis("off")

    plt.tight_layout()

    if run:
        run.log({"Pre transform examples": wandb.Image(fig)})
    else:
        plt.show()

    plt.close(fig)


def log_transforms(
    run: Run,
    batch: Dict,
    grid_size: Tuple[int, int],
    id2lbl: Dict[int, str],
    transforms: Optional[Dict] = None,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> None:
    """
    Logs a grid of transformed images with their bounding boxes to wandb.

    Args:
        run (Run): The wandb run object.
        batch (Tuple[Dict, List]): A single batch of data (processed_batch, targets).
        grid_size (Tuple[int, int], optional): The grid size for displaying images. Defaults to (3, 3).
        id2lbl (Optional[Dict], optional): A dictionary mapping class IDs to labels. Defaults to None.
        transforms (Optional[Dict], optional): The transforms applied. Defaults to None.
        mean (Optional[torch.Tensor], optional): The mean used for normalization. Defaults to None.
        std (Optional[torch.Tensor], optional): The standard deviation used for normalization. Defaults to None.
    """
    images = batch["pixel_values"].detach().clone()
    targets = batch["labels"]
    n_rows, n_cols = grid_size
    max_plots = n_rows * n_cols
    n = min(len(images), max_plots)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
    axes = axes.flatten()

    for ax in axes[n:]:
        ax.axis("off")

    for i in range(n):
        img = images[i]
        # img = unnormalize(img, mean, std).squeeze(0)
        tgt = targets[i]

        img_uint8 = (img.clamp(0, 1) * 255).to(torch.uint8)

        if "boxes" in tgt and "class_labels" in tgt:
            boxes_cywh = tgt["boxes"]

            if boxes_cywh.numel() == 0:
                boxes_to_draw = torch.empty((0, 4), dtype=torch.long)
            else:
                boxes_xyxy = center_to_corners_format(boxes_cywh)
                if boxes_xyxy.max() <= 1.0:
                    h, w = img_uint8.shape[1:]
                    boxes_xyxy[:, [0, 2]] *= w
                    boxes_xyxy[:, [1, 3]] *= h
                boxes_to_draw = boxes_xyxy.long()

            labels = [str(id2lbl[int(lbl)]) for lbl in tgt["class_labels"]]
        else:
            boxes_to_draw = torch.empty((0, 4), dtype=torch.long)
            labels = []

        annotated = draw_bounding_boxes(
            img_uint8,
            boxes=boxes_to_draw,
            labels=np.array(labels),
            colors="red",
            width=2,
            font="fonts/FiraCodeNerdFont-Bold.ttf",
            font_size=30,
        )

        axes[i].imshow(to_pil_image(annotated))
        axes[i].axis("off")

    plt.tight_layout()

    if run:
        run.log({"Post transform examples": wandb.Image(fig)})
        run.log({"transforms": transforms})
    else:
        plt.show()

    plt.close(fig)


def log_training_time(run: Run, start_time: float) -> None:
    """
    Logs the elapsed training time.

    Args:
        run (Run): The wandb run object.
        start_time (float): The start time of training.
    """
    end_time = time.time()
    elapsed = end_time - start_time
    run.log({"training time ": elapsed})


def log_model_params(run: Run, model: nn.Module) -> None:
    """
    Logs the total and trainable parameters of a model.

    Args:
        run (Run): The wandb run object.
        model (nn.Module): The model.
    """
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

    run.log({"total params": total_params, "trainable params": trainable_params})


def log_class_value_counts(
    run: Run, samples: List[Tuple[str, str]], stage: str = "Train"
) -> None:
    """
    Logs the class distribution of a dataset.

    Args:
        run (Run): The wandb run object.
        samples (List[Tuple[Any, Any]]): A list of samples (e.g., [(image, label), ...]).
        stage (str, optional): The dataset stage (e.g., 'Train', 'Test'). Defaults to "Train".
    """
    all_labels = [label for _, label in samples]

    fruit_counts = Counter(all_labels)
    df_counts = pd.DataFrame(
        fruit_counts.items(), columns=["Class", "Count"]
    ).sort_values(by="Count", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.barplot(
        x="Count",
        y="Class",
        data=df_counts,
        ax=ax,
        palette="viridis",
        hue="Class",
        legend=False,
    )

    ax.set_title(f"Class Distribution in {stage.capitalize()} Set")
    plt.tight_layout()

    run.log({f"{stage}_class_distribution": wandb.Image(fig)})
    plt.close(fig)


def log_checkpoint_artifact(
    run: Run, path: str, name: str, epoch: int, wait: bool = False
) -> None:
    """
    Logs a model checkpoint as a wandb artifact.

    Args:
        run (Run): The wandb run object.
        path (str): The path to the checkpoint file.
        name (str): The name of the artifact.
        epoch (int): The epoch number.
        wait (bool, optional): Whether to wait for the artifact to be uploaded. Defaults to False.
    """
    artifact = wandb.Artifact(
        name=f"{name}-checkpoint",
        type="model-checkpoint",
        description=f"Checkpoint at epoch {epoch}",
    )
    artifact.add_file(path)
    run.log_artifact(artifact)
    if wait:
        artifact.wait()


def log_detection_confusion_matrix(
    run: Run, cm_object: ConfusionMatrix, class_names: List[str]
) -> None:
    """
    Logs a detection confusion matrix plot to wandb.

    Args:
        run (Run): The wandb run object.
        cm_object (ConfusionMatrix): The confusion matrix object.
        class_names (List[str]): The list of class names.
    """
    if not run:
        return

    names = class_names.copy()

    fig = cm_object.plot(class_names=names)
    run.log({"val/confusion_matrix": wandb.Image(fig)})
    plt.close(fig)


def log_per_class_metric(
    class_names: List,
    metric_per_class: torch.Tensor,
    ds_type: str,
    metric_name: str,
    log_data: Dict,
) -> None:
    metric_per_class = metric_per_class.detach().cpu()
    for i, name in enumerate(class_names):
        if i < len(metric_per_class):
            log_data[f"{ds_type}/{metric_name}/{name}"] = metric_per_class[i].item()


def log_val_data(epoch: int, best_test_map: float, trainer) -> None:
    """
    Performs final validation, logs metrics, and logs it to wandb

    Args:
        epoch (int): The final epoch number.
        best_test_map (float): The best test mAP@.50 achieved during training.

    Returns:
        None
    """

    trainer.model = trainer.early_stopping.get_best_model(trainer.model)
    val_loss, val_metrics, cm = trainer.eval(trainer.val_dl, epoch + 1, calc_cm=True)

    log_data = {
        "test/best test map": best_test_map,
        "val/map@50:95": val_metrics["map@50:95"],
        "val/map@50": val_metrics["map@50"],
        "val/recall": val_metrics["recall"],
        "val/precision": val_metrics["precision"],
    }

    log_data.update({f"val/{k}": v for k, v in val_loss.items()})

    for metric in ["map@50", "precision", "recall"]:
        log_per_class_metric(
            trainer.val_dl.dataset.labels,
            val_metrics[f"{metric}_per_class"],
            "val",
            metric,
            log_data,
        )
    tqdm.write(
        f"\tVal  --- Loss: {val_loss['loss']:.4f}, mAP50-95: {val_metrics['map@50:95']:.4f}, mAP@50 : {val_metrics['map@50']:.4f}"
    )
    log_detection_confusion_matrix(trainer.run, cm, list(trainer.val_dl.dataset.labels))

    trainer.run.log(log_data)


def log_epoch_data(
    epoch: int,
    train_loss: Dict[str, float],
    test_loss: Dict[str, float],
    test_metrics: Dict,
    trainer,
) -> None:
    """
    Constructs and logs a dictionary of metrics for logging at the end of an epoch.

    Args:
        epoch (int): The current epoch number.
        train_loss (Dict): Dict containing total, classification, bbox and giou training loss for epoch.
        test_map (float): The test mAP@.5-.95.
        test_map50 (float): The test mAP@.50.
        test_loss (Dict): Dict containing total, classification, bbox and giou test loss for epoch.
        test_map_per_class (torch.Tensor): The test mAP@.50 for each class.
        trainer (Trainer) : instance of Trainer class

    Returns:
        None
    """
    log_data = {
        "epoch": epoch,
        "test/map@50:95": test_metrics["map@50:95"],
        "test/map@50": test_metrics["map@50"],
        "test/recall": test_metrics["recall"],
        "test/precision": test_metrics["precision"],
        "Learning rate": float(f"{trainer.scheduler.get_last_lr()[0]:.6f}"),
    }

    log_data.update({f"train/{k}": v for k, v in train_loss.items()})
    log_data.update({f"test/{k}": v for k, v in test_loss.items()})

    for metric in ["map@50", "precision", "recall"]:
        log_per_class_metric(
            trainer.test_dl.dataset.labels,
            test_metrics[f"{metric}_per_class"],
            "test",
            metric,
            log_data,
        )
    trainer.run.log(log_data)
