from collections import Counter
from omegaconf import OmegaConf
import pandas as pd
import torch.nn as nn
from datetime import datetime
import wandb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from .general import unnormalize

def initwandb(cfg):
    name = get_run_name(cfg)
    run = wandb.init(
        entity="mohamedkhayat025-none",
        project="fruit-transformer",
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("test/*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")

    return run


def get_run_name(cfg):
    name = (
        datetime.now().strftime("%Y%m%d-%H%M%S")
        + f"_model={cfg.model.name}_lr={cfg.lr}"
    )
    return name


def log_images(run, batch, id2lbl, grid_size=(3, 3), mean=None, std=None):
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
        run.log({f"Pre transform examples": wandb.Image(fig)})
    else:
        plt.show()

    plt.close(fig)


def log_transforms(run, batch, grid_size, id2lbl, transforms, mean=None, std=None):
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
        run.log({f"Post transform examples": wandb.Image(fig)})
        run.log({f"transforms": transforms})
    else:
        plt.show()

    plt.close(fig)


def log_confusion_matrix(run, y_true, y_pred, classes, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    fmt_string = "d"
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = cm.astype(float) / row_sums
        fmt_string = ".2f"

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt_string,
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.tight_layout()

    run.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)


def log_training_time(run, start_time):
    end_time = time.time()
    elapsed = end_time - start_time
    run.log({"training time ": elapsed})


def log_model_params(run, model: nn.Module):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

    run.log({"total params": total_params, "trainable params": trainable_params})


def log_class_value_counts(run, samples, stage="Train"):
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
