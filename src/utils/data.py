import functools
import pathlib
import re
from .dataset import DS
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from torch.utils.data import Dataset
from .general import seed_worker


def get_samples(root_dir: str, folder: str, debug=False):
    print(f"extracting {folder} samples")
    samples = []
    labels = []

    imgs_folder = pathlib.Path(root_dir, folder)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for img_path in imgs_folder.glob("**/*.jpg"):
        if img_path.is_file() and img_path.suffix.lower() in image_extensions:
            label = img_path.parent.name
            clean_label = re.sub(r"\s+\d+$", "", label)
            labels.append(clean_label)
            samples.append((str(img_path), clean_label))
    labels = list(set(labels))
    return samples, labels


def make_datasets(train_samples, test_samples, labels, id2lbl, lbl2id):
    print("making datasets")

    train_ds = DS(train_samples, labels, id2lbl, lbl2id)
    test_ds = DS(test_samples, labels, id2lbl, lbl2id)

    return train_ds, test_ds


def make_dataloaders(train_ds: Dataset, test_ds: Dataset, cfg: DictConfig):
    print("making dataloaders")
    worker_init = functools.partial(seed_worker, base_seed=cfg.seed)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=worker_init,
    )

    return train_dl, test_dl


def get_labels_and_mappings(train_labels, test_labels):
    labels = sorted(list(set(train_labels + test_labels)))

    id2lbl = {i: lbl for i, lbl in enumerate(labels)}
    lbl2id = {v: k for k, v in id2lbl.items()}

    return labels, id2lbl, lbl2id
