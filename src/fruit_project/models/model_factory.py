# SPDX-FileCopyrightText: 2025 Mohamed Khayat
# SPDX-License-Identifier: AGPL-3.0-or-later

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
    "detrv1_18": "PekingU/rtdetr_r18vd",
    "detrv1_34": "PekingU/rtdetr_r34vd",
    "detrv1_50": "PekingU/rtdetr_r50vd",
    "detrv1_50_365": "PekingU/rtdetr_r50vd_coco_o365",
    "detrv1_101": "PekingU/rtdetr_r101vd",
    # add these models
    # "detr_50": "facebook/detr-resnet-50",
    # "detr_101": "facebook/detr-resnet-101",
    # "detr_50_dc5": "facebook/detr-resnet-50-dc5",
    # "cond_detr_50": "microsoft/conditional-detr-resnet-50",
    "yolos_tiny": "hustvl/yolos-tiny",
    "yolos_small": "hustvl/yolos-small",
    "yolos_base": "hustvl/yolos-base",
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
        return get_hf_model(device, n_classes, id2lbl, lbl2id, cfg)
    else:
        raise ValueError(f"model not supported, use one of : {supported_models}")


def get_hf_model(
    device: torch.device,
    n_classes: int,
    id2label: dict,
    label2id: dict,
    cfg: DictConfig,
) -> Tuple[nn.Module, Compose, List, List, AutoImageProcessor]:
    """
    Loads the HF model along with its configuration, processor, and transformations.

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

    config_kwargs = dict(
        trust_remote_code=True,
        num_labels=n_classes,
        id2label=id2label,
        label2id=label2id,
    )

    if hasattr(cfg.model, "decoder_method"):
        config_kwargs["decoder_method"] = cfg.model.decoder_method

    config = AutoConfig.from_pretrained(
        checkpoint,
        **config_kwargs,
    )

    model_kwargs = {"config": config, "ignore_mismatched_sizes": True}

    if "yolos" in cfg.model.name:
        model_kwargs.update(
            {"attn_implementation": "sdpa", "torch_dtype": torch.float32}
        )

    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        **model_kwargs,
    )

    model = freeze_weights(model, cfg.freeze_backbone, cfg.partially_freeze_backbone)

    processor = AutoImageProcessor.from_pretrained(
        checkpoint, trust_remote_code=True, use_fast=True
    )

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
) -> nn.Module:
    for name, param in model.named_parameters():
        param.requires_grad = True

        if freeze_backbone and name.startswith("model.backbone"):
            param.requires_grad = False
            if partially_freeze_backbone and name.startswith(
                "model.backbone.model.encoder.stages.3"
            ):
                param.requires_grad = True

    return model
