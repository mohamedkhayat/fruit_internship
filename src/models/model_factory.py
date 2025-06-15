from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    AutoConfig,
)
import torch.nn as nn
import timm
from timm.data import resolve_data_config
from .transforms_factory import get_transforms

supported_models = {
    "detrv2_18": "PekingU/rtdetr_v2_r18vd",
    "detrv2_34": "PekingU/rtdetr_v2_r34vd",
    "detrv2_50": "PekingU/rtdetr_v2_r50vd",
    "detrv2_101": "PekingU/rtdetr_v2_r101vd",
}


def get_model(cfg, device, n_classes, id2lbl, lbl2id, debug=False):
    if cfg.model.name in supported_models.keys():
        return get_RTDETRv2(device, n_classes, id2lbl, lbl2id, cfg)
    else:
        raise ValueError(f"model not supported, use one of : {supported_models}")


def get_RTDETRv2(device, n_classes, id2label, label2id, cfg):
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

    processor = AutoImageProcessor.from_pretrained(checkpoint, trust_remote_code=True)

    transforms = get_transforms(cfg.model.input_size)
    print("model loaded")
    return (
        model.to(device),
        transforms,
        processor.image_mean,
        processor.image_std,
        processor,
    )
