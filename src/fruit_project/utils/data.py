# SPDX-FileCopyrightText: 2025 Mohamed Khayat
# SPDX-License-Identifier: AGPL-3.0-or-later

import functools
from tqdm import tqdm
from fruit_project.utils.datasets.alb_mosaic_dataset import (
    create_albumentations_mosaic_dataset,
)
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

    train_ds_base = DET_DS(
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
    return train_ds_base, test_ds, val_ds


def get_sampler(train_ds: DET_DS, generator) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler for the training dataset.
    Handles the new dataset format which returns a single dictionary.
    """
    print("Creating weighted sampler...")
    class_counts: Counter = Counter()
    image_classes = []
    for label_path in tqdm(
        train_ds.label_paths, desc="1/2: Counting classes for sampler"
    ):
        classes_in_image = set()
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.strip().split()[0])
                        classes_in_image.add(class_id)

        class_counts.update(classes_in_image)
        image_classes.append(classes_in_image)

    if not class_counts:
        print(
            "Warning: No classes found in dataset for sampler. Using uniform sampling."
        )
        return None

    class_weights = {c: 1.0 / cnt for c, cnt in class_counts.items()}

    weights = []
    for classes_in_image in tqdm(image_classes, desc="2/2: Assigning Class Weights"):
        if not classes_in_image:
            weights.append(min(class_weights.values()) if class_weights else 1.0)
            continue

        weights.append(max(class_weights[c] for c in classes_in_image))

    weights = torch.tensor(weights, dtype=torch.double)
    sampler_generator = torch.Generator().manual_seed(generator.initial_seed() + 1)

    sampler = WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True, generator=sampler_generator
    )

    print("Weighted sampler created.")

    return sampler


def make_dataloaders(
    cfg: DictConfig,
    train_ds_base: DET_DS,
    test_ds: DET_DS,
    val_ds: DET_DS,
    generator: torch.Generator,
    processor: AutoImageProcessor,
    transforms: Compose,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
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
        Tuple[DataLoader, DataLoader, DataLoader, Tuple[torch.Tensor, torch.Tensor]]: The training, testing, validation dataloaders and a training sample.
    """
    print("making dataloaders")

    worker_init = functools.partial(seed_worker, base_seed=cfg.seed)
    collate = functools.partial(collate_fn)

    for ds in [train_ds_base, test_ds, val_ds]:
        ds.processor = processor

    train_ds_base, test_ds, val_ds = set_transforms(
        train_ds_base, test_ds, val_ds, transforms, cfg
    )
    if cfg.mosaic.use:
        print("Applying Mosaic augmentation to the training dataset.")
        """
        train_ds = create_ultralytics_mosaic_dataset(
            dataset=train_ds_base,
            target_size=train_ds_base.input_size,
            mosaic_prob=cfg.mosaic.prob,
            disable_mosaic_epochs=cfg.mosaic.disable_epoch,
            total_epochs=cfg.epochs,
        )
        """
        train_ds = create_albumentations_mosaic_dataset(
            train_ds_base,
            cfg.model.input_size,
            cfg.mosaic.prob,
            cfg.mosaic.disable_epoch,
            cfg.epochs,
            transforms["train"],
            transforms["train_easy"],
            cfg.min_viz,
            cfg.min_area,
        )
    else:
        train_ds = train_ds_base
    sampler = None
    if cfg.do_sample:
        sampler = get_sampler(train_ds, generator)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.step_batch_size,
        shuffle=not cfg.do_sample,
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
        generator=torch.Generator().manual_seed(generator.initial_seed() + 2),
        collate_fn=collate,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.step_batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(generator.initial_seed() + 3),
        collate_fn=collate,
    )

    test_sample = next(iter(test_dl))
    return train_dl, test_dl, val_dl, test_sample


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


def collate_fn(batch: BatchEncoding) -> Dict:
    """
    Collates a batch of data for the dataloader.

    Args:
        batch (BatchEncoding): A batch of data containing images and targets.

    Returns:
        Tuple[BatchEncoding, List]: Processed batch and list of targets.
    """
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data


def set_transforms(
    train_ds,
    test_ds,
    val_ds,
    transforms: Compose,
    cfg,
):
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

    train_ds.transforms = transforms["train"]

    test_ds.transforms = transforms["test"]
    val_ds.transforms = transforms["test"]

    return train_ds, test_ds, val_ds
