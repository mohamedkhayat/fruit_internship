from collections import Counter
from omegaconf import OmegaConf
import pandas as pd
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from fruit_project.utils.general import unnormalize
from omegaconf import DictConfig
from wandb.sdk.wandb_run import Run
from typing import Dict, Tuple, List, Optional
from fruit_project.utils.metrics import ConfusionMatrix


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
    name = f"model={cfg.model.name}_lr={cfg.lr}"
    return name


def log_images(
    run: Run,
    batch: Tuple[Dict, List],
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
    processed_batch, targets = batch
    images = processed_batch["pixel_values"].detach().clone()
    n_rows, n_cols = grid_size
    max_plots = n_rows * n_cols
    n = min(len(images), max_plots)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
    axes = axes.flatten()

    for ax in axes[n:]:
        ax.axis("off")

    for i in range(n):
        img = images[i]
        img = unnormalize(img, mean, std).squeeze(0)
        tgt = targets[i]

        img_uint8 = (img * 255).to(torch.uint8)

        boxes = []
        labels = []

        for ann in tgt["annotations"]:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(str(id2lbl[int(ann["category_id"])]))

        annotated = draw_bounding_boxes(
            img_uint8,
            boxes=torch.tensor(boxes, dtype=torch.int64),
            labels=labels,
            colors="red",
            width=2,
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
    batch: Tuple[Dict, List],
    grid_size: Tuple[int, int] = (3, 3),
    id2lbl: Optional[Dict] = None,
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
    processed_batch, targets = batch
    images = processed_batch["pixel_values"].detach().clone()
    n_rows, n_cols = grid_size
    max_plots = n_rows * n_cols
    n = min(len(images), max_plots)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
    axes = axes.flatten()

    for ax in axes[n:]:
        ax.axis("off")

    for i in range(n):
        img = images[i]
        img = unnormalize(img, mean, std).squeeze(0)
        tgt = targets[i]

        img_uint8 = (img.clamp(0, 1) * 255).to(torch.uint8)

        boxes = []
        labels = []

        for ann in tgt["annotations"]:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(str(id2lbl[int(ann["category_id"])]))

        annotated = draw_bounding_boxes(
            img_uint8,
            boxes=torch.tensor(boxes, dtype=torch.int64),
            labels=labels,
            colors="red",
            width=2,
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
