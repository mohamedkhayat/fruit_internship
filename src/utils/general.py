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


def is_hf_model(model):
    return isinstance(model, transformers.PreTrainedModel)
