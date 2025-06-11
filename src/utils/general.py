import matplotlib.pyplot as plt
import torch
from timm.data import infer_imagenet_subset, ImageNetInfo
import random
import numpy as np
import timm
from torchvision.transforms.functional import to_pil_image
from transformers import (
    AutoModelForImageClassification,
)
import inspect
import transformers
import timm


def set_seed(SEED):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    return generator


def seed_worker(worker_id, base_seed):
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def plot_img(img, label=None):
    # c x h x w -> h x w x c
    print(img.shape)

    img = img.detach().cpu().numpy()
    plt.title(label if label is not None else "img")
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def unnormalize(img_tensor, mean, std):
    mean = torch.tensor(mean).to(img_tensor.device)
    std = torch.tensor(std).to(img_tensor.device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)

    return img_tensor * std + mean


def check_transforms(
    model, device, test_ds, test_dl, mean, std, model_type, model_trans, alb_trans
):
    # gonna move this to a test folder and make it a unit test
    if model_type == "timm":
        check_timm(model, device, test_ds, test_dl, mean, std, model_trans, alb_trans)
    elif model_type == "hf":
        check_hf(model, device, test_ds, test_dl, mean, std, model_trans, alb_trans)
    elif model_type == "tv":
        check_timm(model, device, test_ds, test_dl, mean, std, model_trans, alb_trans)
    else:
        print(f"error unkown model type : {model_type}")


def check_timm(model, device, test_ds, test_dl, mean, std, timm_config, alb_transforms):
    def id2label(idx):
        return info.index_to_description(idx)

    checkpoint = "vit_tiny_patch16_224.augreg_in21k"
    model = timm.create_model(checkpoint, pretrained=True).to(device)
    timm_transforms = timm.data.create_transform(**timm_config)
    subset = infer_imagenet_subset(model)
    info = ImageNetInfo(subset)
    test_dl.dataset.transforms = None
    model.eval()

    with torch.no_grad():
        imgs, labels = next(iter(test_dl))
        img, label = imgs[0], labels[0]
        print(img.shape)
        timm_tf_img = img.permute(2, 0, 1)
        print(f"timm tf img shape : {timm_tf_img.shape}")
        timm_tf_img = timm_transforms(to_pil_image(timm_tf_img))
        alb_tf_img = alb_transforms(image=img.numpy())["image"]
        timm_img = timm_tf_img.to(device)
        alb_img = alb_tf_img.to(device)
        timm_img = torch.unsqueeze(timm_img, dim=0)
        alb_img = torch.unsqueeze(alb_img, dim=0)
        timm_outputs = model(timm_img)
        alb_outputs = model(alb_img)
        timm_predicted_class_idx = timm_outputs.argmax(-1).item()
        alb_predicted_class_idx = alb_outputs.argmax(-1).item()
        print(f"timm predicted class : {id2label(timm_predicted_class_idx)}")
        print(f"alb predicted class : {id2label(alb_predicted_class_idx)}")
        print(f"label : {test_ds.id2lbl[int(label)]}")


def check_hf(model, device, test_ds, test_dl, mean, std, hf_transforms, alb_transforms):
    checkpoint = "google/vit-base-patch16-224"
    print(f"getting : {checkpoint}")

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
    ).to(device)
    model.eval()
    test_dl.dataset.transforms = None
    with torch.no_grad():
        imgs, labels = next(iter(test_dl))
        img, label = imgs[0], labels[0]
        hf_tf_img = hf_transforms(img, return_tensors="pt")["pixel_values"]
        alb_tf_img = alb_transforms(image=img.numpy())["image"]
        hf_tf_img = hf_tf_img.to(device)
        alb_tf_img = alb_tf_img.to(device)
        alb_tf_img = torch.unsqueeze(alb_tf_img, dim=0)
        hf_outputs = model(hf_tf_img)
        alb_outputs = model(alb_tf_img)
        hf_logits = hf_outputs.logits
        alb_logits = alb_outputs.logits
        hf_predicted_class_idx = hf_logits.argmax(-1).item()
        alb_predicted_class_idx = alb_logits.argmax(-1).item()
        print(f"hf predicted class : {model.config.id2label[hf_predicted_class_idx]}")
        print(f"alb predicted class : {model.config.id2label[alb_predicted_class_idx]}")
        print(f"label : {test_ds.id2lbl[int(label)]}")


def is_hf_model(model):
    return isinstance(model, transformers.PreTrainedModel)
