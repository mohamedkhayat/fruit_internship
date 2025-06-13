from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    AutoConfig,
)
import torch.nn as nn
import timm
from timm.data import resolve_data_config
from .transforms_factory import get_transforms

supported_models = [
    "detrv2",
]


def get_model(cfg, device, n_classes, id2lbl, lbl2id, debug=False):
    if debug == False and cfg.model.name not in supported_models:
        raise ValueError(f"model not supported, use one of : {supported_models}")

    if cfg.model.name == "detrv2":
        return get_RTDETRv2(device, n_classes, id2lbl, lbl2id)


def get_RTDETRv2(device, n_classes, id2label, label2id):
    checkpoint = "PekingU/rtdetr_v2_r18vd"
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

    processor = AutoImageProcessor.from_pretrained(checkpoint, trust_remote_code=True)

    transforms = get_transforms()
    print("model loaded")
    return (
        model.to(device),
        transforms,
        processor.image_mean,
        processor.image_std,
        processor,
    )
