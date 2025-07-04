import functools
import numpy as np
from fruit_project.utils.datasets.det_dataset import DET_DS
from torch.utils.data import DataLoader, WeightedRandomSampler
from omegaconf import DictConfig
from fruit_project.utils.general import seed_worker
import os
from collections import Counter
import torch
from hydra.utils import get_original_cwd
import resource  # <-- Import the resource module
from albumentations import Compose
from typing import Dict, List, Tuple
from transformers import AutoImageProcessor, BatchEncoding

# Increase the number of open file descriptors
try:
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Set the soft limit to the hard limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))
except (ValueError, OSError) as e:
    print(f"Could not set RLIMIT_NOFILE: {e}")


def download_dataset():
    """
    Downloads the dataset from Kaggle using the Kaggle API.

    Raises:
        RuntimeError: If the required environment variables for Kaggle API are not set.

    Returns:
        None
    """
    username = os.getenv("KAGGLE_USERNAME")
    api_key = os.getenv("KAGGLE_KEY")
    if api_key is None or username is None:
        raise RuntimeError(
            "Environment variable 'kaggle_key' and or 'username' is not set!"
        )

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = api_key

    from kaggle import api

    api.authenticate()
    print("download dataset")
    api.dataset_download_files(
        "lakshaytyagi01/fruit-detection", path="./data", unzip=True
    )


def make_datasets(cfg: DictConfig) -> Tuple[DET_DS, DET_DS, DET_DS]:
    """
    Creates training, testing, and validation datasets.

    Args:
        cfg (DictConfig): Configuration object containing dataset parameters.

    Returns:
        Tuple[DET_DS, DET_DS, DET_DS]: The training, testing, and validation datasets.
    """
    if cfg.download_data:
        download_dataset()

    print("making datasets")
    data_dir = os.path.join(get_original_cwd(), "data", cfg.root_dir)

    train_ds = DET_DS(
        data_dir,
        "train",
        "images",
        "labels",
        "data.yaml",
        None,
        cfg.model.input_size,
    )
    test_ds = DET_DS(
        data_dir,
        "test",
        "images",
        "labels",
        "data.yaml",
        None,
        cfg.model.input_size,
    )
    val_ds = DET_DS(
        data_dir,
        "val",
        "images",
        "labels",
        "data.yaml",
        None,
        cfg.model.input_size,
    )
    return train_ds, test_ds, val_ds


def get_sampler(train_ds: DET_DS, strat: str) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler for the training dataset.

    Args:
        train_ds (DET_DS): The training dataset.
        strat (str): The strategy for weighting ('max' or 'mean').

    Returns:
        WeightedRandomSampler: A sampler for the training dataset.
    """
    class_counts: Counter = Counter()
    for _, target in train_ds:
        classes = {ann["category_id"] for ann in target["annotations"]}
        class_counts.update(classes)

    class_weights = {c: 1.0 / cnt for c, cnt in class_counts.items()}

    weights = []
    for _, target in train_ds:
        classes = {ann["category_id"] for ann in target["annotations"]}
        if not classes:
            weights.append(min(class_weights.values()))
            continue
        if strat == "max":
            weights.append(max(class_weights[c] for c in classes))  # maybe try mean
        elif strat == "mean":
            weights.append(np.mean([class_weights[c] for c in classes]))

    weights = torch.tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    return sampler


def make_dataloaders(
    cfg: DictConfig,
    train_ds: DET_DS,
    test_ds: DET_DS,
    val_ds: DET_DS,
    generator: torch.Generator,
    processor: AutoImageProcessor,
    transforms: Compose,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates dataloaders for training, testing, and validation datasets.

    Args:
        cfg (DictConfig): Configuration object containing dataloader parameters.
        train_ds (DET_DS): The training dataset.
        test_ds (DET_DS): The testing dataset.
        val_ds (DET_DS): The validation dataset.
        generator (torch.Generator): A PyTorch generator for reproducibility.
        processor (AutoImageProcessor): Processor for image preprocessing.
        transforms (Compose): Transformations to apply to the datasets.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The training, testing, and validation dataloaders.
    """
    print("making dataloaders")

    worker_init = functools.partial(seed_worker, base_seed=cfg.seed)
    collate = functools.partial(collate_fn, processor=processor)

    sampler = get_sampler(train_ds, cfg.sample_strat)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.step_batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init,
        generator=generator,
        collate_fn=collate,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=cfg.step_batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=worker_init,
        generator=generator,
        collate_fn=collate,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.step_batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=worker_init,
        generator=generator,
        collate_fn=collate,
    )
    train_dl, test_dl, val_dl = set_transforms(train_dl, test_dl, val_dl, transforms)
    return train_dl, test_dl, val_dl


def get_labels_and_mappings(
    train_labels: List, test_labels: List
) -> Tuple[List, Dict, Dict]:
    """
    Generates labels and mappings for class IDs and names.

    Args:
        train_labels (List): List of labels from the training dataset.
        test_labels (List): List of labels from the testing dataset.

    Returns:
        Tuple[List, Dict, Dict]: A tuple containing:
            - labels (List): Sorted list of unique labels.
            - id2lbl (Dict): Mapping from class IDs to labels.
            - lbl2id (Dict): Mapping from labels to class IDs.
    """
    labels = sorted(list(set(train_labels + test_labels)))

    id2lbl = {i: lbl for i, lbl in enumerate(labels)}
    lbl2id = {v: k for k, v in id2lbl.items()}

    return labels, id2lbl, lbl2id


def collate_fn(
    batch: BatchEncoding, processor: AutoImageProcessor
) -> Tuple[BatchEncoding, List]:
    """
    Collates a batch of data for the dataloader.

    Args:
        batch (BatchEncoding): A batch of data containing images and targets.
        processor (AutoImageProcessor): Processor for image preprocessing.

    Returns:
        Tuple[BatchEncoding, List]: Processed batch and list of targets.
    """
    imgs, targets = list(zip(*batch))
    batch_processed = processor(
        images=imgs,
        annotations=targets,
        return_tensors="pt",
        do_resize=False,
        do_pad=False,
        do_normalize=True,
    )
    return (batch_processed, list(targets))


def set_transforms(
    train_dl: DataLoader, test_dl: DataLoader, val_dl: DataLoader, transforms: Compose
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Sets transformations for the datasets in the dataloaders.

    Args:
        train_dl (DataLoader): Training dataloader.
        test_dl (DataLoader): Testing dataloader.
        val_dl (DataLoader): Validation dataloader.
        transforms (Compose): Transformations to apply.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Updated dataloaders with transformations applied.
    """
    train_dl.dataset.transforms = transforms["train"]
    test_dl.dataset.transforms = transforms["test"]
    val_dl.dataset.transforms = transforms["test"]

    return train_dl, test_dl, val_dl
