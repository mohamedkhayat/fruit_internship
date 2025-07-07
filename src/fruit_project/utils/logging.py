from collections import Counter
import numpy as np
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
    batch: Tuple[Dict, List],
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


def log_confidence_analysis(
    run: Run, cm_object: ConfusionMatrix, class_names: List[str]
) -> None:
    """
    Logs confidence score analysis for debugging detection issues.

    Args:
        run (Run): The wandb run object.
        cm_object (ConfusionMatrix): The confusion matrix object with confidence tracking.
        class_names (List[str]): The list of class names.
    """
    if not run or not hasattr(cm_object, 'confidence_scores'):
        return

    try:
        # Get confidence analysis
        analysis = cm_object.get_confidence_analysis(class_names)
        
        # Log overall statistics
        if 'overall' in analysis:
            overall_stats = analysis['overall']
            run.log({
                "confidence_analysis/overall_mean": overall_stats['mean_confidence'],
                "confidence_analysis/overall_std": overall_stats['std_confidence'],
                "confidence_analysis/total_detections": overall_stats['total_detections']
            })
        
        # Log per-class statistics
        for class_name, stats in analysis.items():
            if class_name == 'overall':
                continue
                
            class_id = stats['class_id']
            prefix = f"confidence_analysis/{class_name}"
            
            # Log detection metrics
            run.log({
                f"{prefix}/precision": stats['precision'],
                f"{prefix}/recall": stats['recall'],
                f"{prefix}/tp": stats['tp'],
                f"{prefix}/fp": stats['fp'], 
                f"{prefix}/fn": stats['fn'],
                f"{prefix}/total_detections": stats['total_detections'],
                f"{prefix}/correct_detections": stats['correct_detections'],
                f"{prefix}/incorrect_detections": stats['incorrect_detections'],
                f"{prefix}/above_threshold_detections": stats['above_threshold_detections'],
                f"{prefix}/current_threshold": stats['current_threshold']
            })
            
            # Log confidence statistics
            conf_stats = stats['confidence_stats']
            run.log({
                f"{prefix}/conf_all_mean": conf_stats['all_mean'],
                f"{prefix}/conf_all_std": conf_stats['all_std'],
                f"{prefix}/conf_correct_mean": conf_stats['correct_mean'],
                f"{prefix}/conf_correct_std": conf_stats['correct_std'],
                f"{prefix}/conf_incorrect_mean": conf_stats['incorrect_mean'],
                f"{prefix}/conf_incorrect_std": conf_stats['incorrect_std']
            })
            
            # Log suggested thresholds if available
            if 'suggested_thresholds' in stats:
                thresholds = stats['suggested_thresholds']
                run.log({
                    f"{prefix}/suggested_threshold_p25": thresholds['p25'],
                    f"{prefix}/suggested_threshold_p50": thresholds['p50'],
                    f"{prefix}/suggested_threshold_p75": thresholds['p75']
                })
        
        # Create and log confidence distribution plots
        try:
            fig = cm_object.plot_confidence_distributions(class_names)
            run.log({"confidence_analysis/distributions": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not create confidence distribution plot: {e}")
            
    except Exception as e:
        print(f"Warning: Could not log confidence analysis: {e}")


def log_detailed_class_stats(
    run: Run, cm_object: ConfusionMatrix, class_names: List[str]
) -> None:
    """
    Logs detailed per-class detection statistics for debugging.

    Args:
        run (Run): The wandb run object.
        cm_object (ConfusionMatrix): The confusion matrix object.
        class_names (List[str]): The list of class names.
    """
    if not run:
        return
    
    # Create a summary table for wandb
    table_data = []
    for class_id in range(len(class_names)):
        if class_id >= cm_object.nc:
            break
            
        stats = cm_object.class_stats[class_id]
        class_name = class_names[class_id]
        
        precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
        recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Get confidence stats if available
        class_confidences = [entry['confidence'] for entry in cm_object.confidence_scores 
                           if entry['predicted_class'] == class_id]
        avg_confidence = np.mean(class_confidences) if class_confidences else 0
        
        table_data.append([
            class_name,
            stats['tp'],
            stats['fp'], 
            stats['fn'],
            stats['total_gt'],
            stats['total_detections'],
            f"{precision:.3f}",
            f"{recall:.3f}", 
            f"{f1_score:.3f}",
            f"{avg_confidence:.3f}",
            f"{cm_object._get_conf_threshold(class_id):.3f}"
        ])
    
    # Create wandb table
    table = wandb.Table(
        columns=[
            "Class", "TP", "FP", "FN", "Total GT", "Total Det", 
            "Precision", "Recall", "F1", "Avg Conf", "Threshold"
        ],
        data=table_data
    )
    
    run.log({"confidence_analysis/detailed_stats": table})
