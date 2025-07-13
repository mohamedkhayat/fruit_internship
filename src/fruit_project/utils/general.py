# SPDX-FileCopyrightText: 2025 Mohamed Khayat
# SPDX-License-Identifier: AGPL-3.0-or-later

import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import transformers
from typing import Optional


def set_seed(SEED: int) -> torch.Generator:
    """
    Sets the seed for reproducibility across various libraries.

    Args:
        SEED (int): The seed value to use.

    Returns:
        torch.Generator: A PyTorch generator seeded with the given value.
    """
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    return generator


def seed_worker(worker_id: int, base_seed: int) -> None:
    """
    Seeds a worker for multiprocessing to ensure reproducibility.

    Args:
        worker_id (int): The ID of the worker.
        base_seed (int): The base seed value.

    Returns:
        None
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def plot_img(img, label: Optional[str] = None) -> None:
    """
    Plots an image using matplotlib.

    Args:
        img (torch.Tensor): The image tensor to plot (shape: C x H x W).
        label (str, optional): The label to display as the title. Defaults to None.

    Returns:
        None
    """
    # c x h x w -> h x w x c
    print(img.shape)

    img = img.detach().cpu().numpy()
    plt.title(label if label is not None else "img")
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def unnormalize(
    img_tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """
    Unnormalizes an image tensor by reversing normalization.

    Args:
        img_tensor (torch.Tensor): The normalized image tensor (shape: N x C x H x W or C x H x W).
        mean (torch.Tensor): The mean used for normalization.
        std (torch.Tensor): The standard deviation used for normalization.

    Returns:
        torch.Tensor: The unnormalized image tensor.
    """
    mean = torch.tensor(mean).to(img_tensor.device)
    std = torch.tensor(std).to(img_tensor.device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)

    return img_tensor * std + mean


def is_hf_model(model) -> bool:
    """
    Checks if the given model is a Hugging Face PreTrainedModel.

    Args:
        model: The model to check.

    Returns:
        bool: True if the model is a Hugging Face PreTrainedModel, False otherwise.
    """
    return isinstance(model, transformers.PreTrainedModel)
