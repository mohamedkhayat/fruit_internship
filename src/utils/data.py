import functools
import pathlib
import re
from .datasets.det_dataset import DET_DS
from torch.utils.data import DataLoader, WeightedRandomSampler
from omegaconf import DictConfig
from .general import seed_worker
import os
from collections import Counter
import torch
from hydra.utils import get_original_cwd
import resource  # <-- Import the resource module

# Increase the number of open file descriptors
try:
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Set the soft limit to the hard limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))
except (ValueError, OSError) as e:
    print(f"Could not set RLIMIT_NOFILE: {e}")


ACCEPTED_LABELS = [
    "apple",
    "cherry",
    "fig",
    "olive",
    "pomegranate",
    "orange",
    "watermelon",
    "strawberry",
    "potato",
    "tomato",
    "pepper",
]


def download_dataset():
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


def get_samples(root_dir: str, folder: str, debug=False):
    print(f"extracting {folder} samples")
    samples = []
    labels = []

    imgs_folder = pathlib.Path(root_dir, folder)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for img_path in imgs_folder.glob("**/*.jpg"):
        if img_path.is_file() and img_path.suffix.lower() in image_extensions:
            label = img_path.parent.name
            clean_label = re.sub(r"\s+\d+$", "", label).split()[0].lower()
            if clean_label in ACCEPTED_LABELS:
                labels.append(clean_label)
                samples.append((str(img_path), clean_label))
    labels = list(set(labels))
    return samples, labels


def make_datasets(cfg):
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
        "valid",
        "images",
        "labels",
        "data.yaml",
        None,
        cfg.model.input_size,
    )

    return train_ds, test_ds, val_ds


def get_sampler(train_ds):
    class_counts = Counter()
    for _, target in train_ds:
        classes = {ann["category_id"] for ann in target["annotations"]}
        class_counts.update(classes)

    class_weights = {c: 1.0 / cnt for c, cnt in class_counts.items()}

    weights = []
    for _, target in train_ds:
        classes = {ann["category_id"] for ann in target["annotations"]}
        weights.append(max(class_weights[c] for c in classes))  # maybe try mean

    weights = torch.tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    return sampler


def make_dataloaders(
    cfg: DictConfig, train_ds, test_ds, val_ds, generator, processor, transforms
):
    print("making dataloaders")

    worker_init = functools.partial(seed_worker, base_seed=cfg.seed)
    collate = functools.partial(collate_fn, processor=processor)

    sampler = get_sampler(train_ds)

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


def get_labels_and_mappings(train_labels, test_labels):
    labels = sorted(list(set(train_labels + test_labels)))

    id2lbl = {i: lbl for i, lbl in enumerate(labels)}
    lbl2id = {v: k for k, v in id2lbl.items()}

    return labels, id2lbl, lbl2id


def collate_fn(batch, processor):
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
    train_dl: DataLoader, test_dl: DataLoader, val_dl: DataLoader, transforms
):
    train_dl.dataset.transforms = transforms["train"]
    test_dl.dataset.transforms = transforms["test"]
    val_dl.dataset.transforms = transforms["test"]

    return train_dl, test_dl, val_dl
