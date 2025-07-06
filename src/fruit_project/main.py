import glob
from omegaconf import DictConfig
import hydra
from fruit_project.utils.data import make_datasets, make_dataloaders
from fruit_project.models.model_factory import get_model
import torch
import cv2
from fruit_project.utils.trainer import Trainer
from fruit_project.utils.logging import (
    initwandb,
    get_run_name,
    log_transforms,
    log_images,
)
from fruit_project.utils.general import set_seed

cv2.setNumThreads(0)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
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

    train_ds, test_ds, val_ds = make_datasets(cfg)

    model, transforms, mean, std, processor = get_model(
        cfg,
        device,
        len(train_ds.labels),
        train_ds.id2lbl,
        train_ds.lbl2id,
    )

    train_dl, test_dl, val_dl, test_sample = make_dataloaders(
        cfg, train_ds, test_ds, val_ds, generator, processor, transforms
    )

    log_images(run, test_sample, test_ds.id2lbl, (3, 3), mean, std)
    log_transforms(
        run, next(iter(train_dl)), (3, 3), train_ds.id2lbl, transforms, mean, std
    )

    trainer = Trainer(
        model, processor, device, cfg, name, run, train_dl, test_dl, val_dl
    )
    print("Setup complete.")

    if cfg.load_ckpt:
        existing = sorted(glob.glob("../checkpoints/*.pth"))
        if existing:
            trainer._load_checkpoint(existing[-1])

    trainer.fit()


if __name__ == "__main__":
    main()
