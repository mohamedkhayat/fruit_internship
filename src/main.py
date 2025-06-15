from omegaconf import DictConfig
import hydra
from utils.data import make_datasets, make_dataloaders, download_dataset, set_transforms
from models.model_factory import get_model
import torch
import cv2
from utils.trainer import Trainer
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
    if cfg.log:
        run = initwandb(cfg)
        name = run.name
    else:
        run = None
        name = get_run_name(cfg)

    generator = set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"using : {device}")

    train_dl, test_dl, val_dl = make_dataloaders(cfg, generator)

    model, transforms, mean, std, processor = get_model(
        cfg,
        device,
        len(train_dl.dataset.labels),
        train_dl.dataset.id2lbl,
        train_dl.dataset.lbl2id,
    )

    train_dl, test_dl, val_dl = set_transforms(train_dl, test_dl, val_dl, transforms)

    log_images(run, next(iter(test_dl)), test_dl.dataset.id2lbl)
    log_transforms(
        run, next(iter(train_dl)), (3, 3), train_dl.dataset.id2lbl, transforms
    )

    trainer = Trainer(
        model, processor, device, cfg, name, run, train_dl, test_dl, val_dl
    )
    print("Setup complete.")

    trainer.fit()


if __name__ == "__main__":
    main()
