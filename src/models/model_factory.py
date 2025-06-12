from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoConfig,
)
from torchvision.models import (
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
)
import torch.nn as nn
import timm
from timm.data import resolve_data_config
from .transforms_factory import get_transforms

supported_models = ["vit_base", "tiny_vit", "efficientnet_v2_s", "efficientnet_b0"]


def get_model(cfg, device, n_classes, id2lbl, lbl2id, debug=False):
    if debug == False and cfg.model.name not in supported_models:
        raise ValueError(f"model not supported, use one of : {supported_models}")

    if cfg.model.name == "vit_base":
        return get_vit_base(device, n_classes, id2lbl, lbl2id)

    elif cfg.model.name == "tiny_vit":
        return get_tiny_vit(device, n_classes)

    elif cfg.model.name == "efficientnet_v2_s":
        return get_efficientnet_v2_s(device, n_classes)

    elif cfg.model.name == "efficientnet_b0":
        return get_efficientnet_b0(device, n_classes)


def get_tiny_vit(device, n_classes):
    checkpoint = "vit_tiny_patch16_224.augreg_in21k"
    print(f"getting : {checkpoint}")

    model = timm.create_model(checkpoint, pretrained=True, num_classes=n_classes)
    processor = resolve_data_config({}, model=model)

    transforms = get_transforms(processor, "tiny_vit")
    print("model loaded")
    return model.to(device), transforms, processor["mean"], processor["std"], processor


def get_vit_base(device, n_classes, id2label, label2id):
    checkpoint = "google/vit-base-patch16-224"
    print(f"getting : {checkpoint}")

    config = AutoConfig.from_pretrained(
        checkpoint, num_labels=n_classes, id2label=id2label, label2id=label2id
    )
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
    )

    processor = AutoImageProcessor.from_pretrained(checkpoint)

    transforms = get_transforms(processor, "vit_base")
    print("model loaded")
    return (
        model.to(device),
        transforms,
        processor.image_mean,
        processor.image_std,
        processor,
    )


def get_efficientnet_v2_s(device, n_classes):
    print(f"getting : efficientnet_v2_s")

    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    model = freeze_params(model, "efficientnet_v2_s")

    transforms = get_transforms(weights, "efficientnet_v2_s")
    print("model loaded")
    return (
        model.to(device),
        transforms,
        weights.transforms().mean,
        weights.transforms().std,
        weights.transforms(),
    )


def get_efficientnet_b0(device, n_classes):
    print(f"getting : efficientnet_b0")

    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    model = freeze_params(model, "efficientnet_b0")

    transforms = get_transforms(weights, "efficientnet_b0")
    print("model loaded")
    return (
        model.to(device),
        transforms,
        weights.transforms().mean,
        weights.transforms().std,
        weights.transforms(),
    )


def freeze_params(model, name):
    if name == "efficientnet_v2_s":
        print("frozen backbone partially")
        return freeze_efficientnet_v2_s(model)
    elif name == "efficientnet_b0":
        print("frozen backbone partially")
        return freeze_efficientnet_b0(model)
    else:
        print("error freezing weigths")


def freeze_efficientnet_v2_s(model):
    grad = False
    for param in model.parameters():
        param.requires_grad = grad

    for name, param in model.named_parameters():
        if "features.6.14" in name:
            grad = True

        param.requires_grad_(grad)

    return model


def freeze_efficientnet_b0(model):
    grad = False
    for param in model.parameters():
        param.requires_grad = grad

    for name, param in model.named_parameters():
        if "features.7" in name:
            grad = True

        param.requires_grad_(grad)

    return model
