from omegaconf import DictConfig
import hydra
from utils.data import make_datasets, make_dataloaders, download_dataset, set_transforms
from models.model_factory import get_model
import torch
import torch.nn as nn
import cv2
from utils.trainer import Trainer
from tqdm import tqdm
from utils.logging import (
    initwandb,
    get_run_name,
    log_transforms,
    log_images,
)
from utils.general import set_seed

cv2.setNumThreads(0)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.download_data:
        download_dataset()

    if cfg.log:
        run = initwandb(cfg)
        name = run.name
    else:
        run = None
        name = get_run_name(cfg)

    generator = set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"using : {device}")

    train_ds, test_ds, val_ds = make_datasets(cfg)
    train_dl, test_dl, val_dl = make_dataloaders(
        train_ds, test_ds, val_ds, cfg, generator
    )

    model, transforms, mean, std, processor = get_model(
        cfg,
        device,
        len(train_ds.labels),
        train_ds.id2lbl,
        train_ds.lbl2id,
    )  # type:ignore

    train_ds, test_ds, val_ds = set_transforms(train_ds, test_ds, val_ds, transforms)

    log_images(run, next(iter(test_dl)), test_ds.id2lbl)
    log_transforms(run, next(iter(train_dl)), (3, 3), train_ds.id2lbl, transforms)

    trainer = Trainer(
        model, processor, device, cfg, name, run, train_dl, test_dl, val_dl
    )
    print("Setup complete.")

    trainer.fit()


if __name__ == "__main__":
    main()
