import torch
import torch.nn as nn
from fruit_project.models.transforms_factory import get_transforms
from omegaconf import DictConfig
from typing import Dict, List, Tuple
from albumentations import Compose
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    AutoConfig,
)

supported_models = {
    "detrv2_18": "PekingU/rtdetr_v2_r18vd",
    "detrv2_34": "PekingU/rtdetr_v2_r34vd",
    "detrv2_50": "PekingU/rtdetr_v2_r50vd",
    "detrv2_101": "PekingU/rtdetr_v2_r101vd",
}


def get_model(
    cfg: DictConfig, device: torch.device, n_classes: int, id2lbl: Dict, lbl2id: Dict
) -> Tuple[nn.Module, Compose, List, List, AutoImageProcessor]:
    """
    Retrieves and initializes a model based on the provided configuration.
    Args:
        cfg (DictConfig): Configuration object containing model specifications.
        device (torch.device): The device on which the model will be loaded (e.g., 'cpu' or 'cuda').
        n_classes (int): Number of classes for the model's output.
        id2lbl (dict): Mapping from class IDs to labels.
        lbl2id (dict): Mapping from labels to class IDs.
    Returns:
        torch.nn.Module: The initialized model.
    Raises:
        ValueError: If the specified model name in the configuration is not supported.
    """

    if cfg.model.name in supported_models.keys():
        return get_RTDETRv2(device, n_classes, id2lbl, lbl2id, cfg)
    else:
        raise ValueError(f"model not supported, use one of : {supported_models}")


def get_RTDETRv2(
    device: torch.device,
    n_classes: int,
    id2label: dict,
    label2id: dict,
    cfg: DictConfig,
) -> Tuple[nn.Module, Compose, List, List, AutoImageProcessor]:
    """
    Loads the RT-DETRv2 model along with its configuration, processor, and transformations.

    Args:
        device (str): The device to load the model onto (e.g., 'cpu', 'cuda').
        n_classes (int): The number of classes for the object detection task.
        id2label (dict): A dictionary mapping class IDs to class labels.
        label2id (dict): A dictionary mapping class labels to class IDs.
        cfg (object): Configuration object containing model settings, including the model name.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The loaded RT-DETRv2 model moved to the specified device.
            - transforms (callable): The transformation function for preprocessing input images.
            - image_mean (list): The mean values used for image normalization.
            - image_std (list): The standard deviation values used for image normalization.
            - processor (AutoImageProcessor): The processor for handling image inputs.
    """
    checkpoint = supported_models[cfg.model.name]
    print(f"getting : {checkpoint}")

    config = AutoConfig.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        num_labels=n_classes,
        id2label=id2label,
        label2id=label2id,
    )

    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        config=config,
        ignore_mismatched_sizes=True,
    )

    model = freeze_weights(
        model, cfg.freeze_backbone, cfg.partially_freeze_backbone, cfg.freeze_encoder
    )

    processor = AutoImageProcessor.from_pretrained(checkpoint, trust_remote_code=True)

    transforms = get_transforms(cfg, id2label)

    print("model loaded")

    return (
        model.to(device),
        transforms,
        processor.image_mean,
        processor.image_std,
        processor,
    )


def freeze_weights(
    model: nn.Module,
    freeze_backbone=True,
    partially_freeze_backbone=False,
    freeze_encoder=False,
) -> nn.Module:
    for name, param in model.named_parameters():
        if (
            freeze_backbone
            and name.startswith("model.backbone")
            and not (
                partially_freeze_backbone
                and name.startswith("model.backbone.model.encoder.stages.3.layers")
            )
        ):
            param.requires_grad = False

        elif freeze_encoder and name.startswith("model.encoder"):
            param.requires_grad = False

        else:
            param.requires_grad = True

    return model
