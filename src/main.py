import pathlib
from omegaconf import DictConfig
import hydra
from utils.data import (
    get_samples,
    make_datasets,
    make_dataloaders,
    get_labels_and_mappings,
)
from models.model_factory import get_model
import torch
from utils.trainer import eval
import torch.nn as nn
from torch.amp import GradScaler
import cv2
from utils.trainer import train, eval, get_optimizer
from utils.early_stop import EarlyStopping
from tqdm import tqdm
from utils.logging import (
    initwandb,
    get_run_name,
    log_confusion_matrix,
    log_model_params,
    log_transforms,
    log_class_value_counts,
    log_images,
)
from utils.general import set_seed
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

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

    train_ds, test_ds, val_ds = make_datasets(cfg)
    train_dl, test_dl, val_dl = make_dataloaders(
        train_ds, test_ds, val_ds, cfg, generator
    )

    model, transforms, mean, std, model_trans = get_model(
        cfg, device, len(train_ds.labels), train_ds.id2lbl, train_ds.lbl2id
    )  # type:ignore

    train_ds.transforms = transforms["train"]
    test_ds.transforms = transforms["test"]
    val_ds.transforms = transforms["test"]

    log_images(run, next(iter(test_dl)), test_ds.id2lbl)
    log_transforms(run, next(iter(train_dl)), (3, 3), train_ds.id2lbl)

    early_stopping = EarlyStopping(cfg.patience, cfg.delta, "checkpoints", name)

    optimizer = get_optimizer(model, cfg.lr, cfg.lr / 10, cfg.weight_decay)
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs
    )

    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=1e-6
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[cfg.warmup_epochs],
    )
    scaler = GradScaler("cuda")

    print("Setup complete.")
    epoch_pbar = tqdm(total=cfg.epochs, desc="Epochs", position=0, leave=True)

    best_test_map = 0

    for epoch in range(cfg.epochs):
        epoch_pbar.set_description(f"Epoch {epoch + 1}/{cfg.epochs}")

        train_loss = train(
            model,
            device,
            train_dl,
            scaler,
            optimizer,
            epoch + 1,
            model_trans,
        )
        tqdm.write(f"\tTrain --- Loss: {train_loss:.4f}")

        test_loss, test_map, test_map50 = eval(
            model, device, test_dl, epoch + 1, model_trans
        )
        tqdm.write(
            f"\tEval  --- Loss: {test_loss:.4f}, mAP50-95: {test_map:.4f}, mAP@50 : {test_map50:.4f}"
        )

        scheduler.step()

        epoch_pbar.set_postfix_str(
            f"Test Loss: {test_loss:.4f}, Test mAP: {test_map:.4f} , Test mAP@50 : {test_map50:.4f}"
        )

        epoch_pbar.update(1)

        best_test_map = max(test_map50, best_test_map)

        if cfg.log:
            run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "test/map": test_map,
                    "test/map 50": test_map50,
                    "test/loss": test_loss,
                    "Learning rate": float(f"{scheduler.get_last_lr()[0]:.6f}"),
                },
                step="epoch",
            )
        if early_stopping(test_map, model):
            tqdm.write(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    print("Training finished.")

    if cfg.log:
        run.log({"best test map": best_test_map})
        # log_confusion_matrix(run, y_true, y_pred, labels)
        model = early_stopping.get_best_model(model)

        val_loss, val_map, val_map50 = eval(
            model, device, val_dl, epoch + 1, model_trans
        )

        run.log({"val/loss": val_loss, "val/map": val_map, "val/map@50": val_map50})
        run.finish()
    tqdm.write(
        f"\tVal  --- Loss: {val_loss:.4f}, mAP50-95: {val_map:.4f}, mAP@50 : {val_map50:.4f}"
    )

    epoch_pbar.close()


if __name__ == "__main__":
    main()
