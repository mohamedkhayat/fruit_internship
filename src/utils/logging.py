from omegaconf import OmegaConf
import torch
import torch.nn as nn
from datetime import datetime
import wandb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from .general import unnormalize


def initwandb(cfg):
    name = get_run_name(cfg)
    run = wandb.init(
        entity="mohamedkhayat025-none",
        project="fruit-transformer",
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run


def get_run_name(cfg):
    name = (
        datetime.now().strftime("%Y%m%d-%H%M%S")
        + f"_dataset={cfg.root_dir}"
        + f"_model={cfg.model.name}_lr={cfg.lr}"
    )
    return name


def log_transforms(run, batch, n_images, classes, aug, mean, std):
    cols = 3
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(rows * 3, cols * 3))
    axes = axes.flatten()

    images, labels = batch
    fig.suptitle(f"{aug} transforms", fontsize=16)

    for ax, img_tensor, label in zip(axes, images[:n_images], labels[:n_images]):
        # 1 x 3 x H x W
        # img = img.squeeze(0)
        img = img_tensor.detach().clone()
        img = unnormalize(img, mean, std).squeeze(0).cpu().numpy()
        img = img.transpose(1, 2, 0)
        img = (img * 255).clip(0, 255).astype("uint8")

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{classes[label]}")

    plt.tight_layout()
    run.log({"transforms visualization": wandb.Image(fig)})
    plt.close(fig)


def log_confusion_matrix(run, y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
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

    run.log({"total parmas": total_params, "trainable params": trainable_params})
