from omegaconf import DictConfig
import hydra
from utils.data import (
    get_samples,
    make_datasets,
    make_dataloaders,
    get_labels_and_mappings,
)
from utils.general import plot_img, unnormalize, check_transforms
from models.model_factory import get_model
import torch
from utils.trainer import eval
import torch.nn as nn
from torch.optim import AdamW
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
    log_training_time,
    log_transforms,
)
from utils.general import set_seed
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR, LinearLR

cv2.setNumThreads(0)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.log:
        run = initwandb(cfg)
        name = run.name
    else:
        name = get_run_name(cfg)

    generator = set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"using : {device}")

    train_samples, train_labels = get_samples(cfg.root_dir, "Training")
    test_samples, test_labels = get_samples(cfg.root_dir, "Test")

    labels, id2lbl, lbl2id = get_labels_and_mappings(train_labels, test_labels)

    train_ds, test_ds = make_datasets(
        train_samples, test_samples, labels, id2lbl, lbl2id
    )

    train_dl, test_dl = make_dataloaders(train_ds, test_ds, cfg, generator)
    n_classes = len(labels)

    model, transforms, mean, std, model_trans = get_model(
        cfg, device, n_classes, id2lbl, lbl2id
    )  # type:ignore

    """
    if cfg.model.name == "tiny_vit":
        model_type = "timm"
    elif cfg.model.name == "vit_base":
        model_type = "hf"
    else:
        model_type = "tv"

    check_transforms(
        model,
        device,
        test_ds,
        test_dl,
        mean,
        std,
        model_type,
        model_trans,
        transforms["test"],
    )
    import sys
    sys.exit()
    """

    train_ds.transforms = transforms["train"]
    test_ds.transforms = transforms["test"]

    if cfg.log:
        log_model_params(run, model)
        log_transforms(
            run, next(iter(train_dl)), cfg.n_images, train_ds.labels, cfg.aug, mean, std
        )

    early_stopping = EarlyStopping(cfg.patience, cfg.delta, "checkpoints", name)

    criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(model, cfg.lr, cfg.lr / 100, cfg.weight_decay, cfg.freeze)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs
    )
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.warmup_epochs],
    )

    scaler = GradScaler("cuda")

    print("Setup complete.")

    epoch_pbar = tqdm(range(cfg.epochs), desc="Epochs", position=0, leave=True)

    best_val_f1 = 0

    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch + 1}/{cfg.epochs}")

        train_loss, train_f1 = train(
            model, device, train_dl, criterion, scaler, n_classes, optimizer, epoch + 1
        )
        tqdm.write(
            f"Epoch {epoch + 1} Train --- Loss: {train_loss:.4f}, F1: {train_f1:.4f}"
        )

        test_loss, test_f1 = eval(
            model, device, test_dl, criterion, n_classes, epoch + 1
        )
        tqdm.write(
            f"Epoch {epoch + 1} Eval  --- Loss: {test_loss:.4f}, F1: {test_f1:.4f}"
        )

        scheduler.step()

        epoch_pbar.set_postfix_str(f"Val Loss: {test_loss:.4f}, Val F1: {test_f1:.4f}")

        best_val_f1 = max(test_f1, best_val_f1)
        if cfg.log:
            run.log(
                {
                    "train f1": train_f1,
                    "train loss": train_loss,
                    "val f1": test_f1,
                    "val loss": test_loss,
                    "Learning rate": float(f"{scheduler.get_last_lr()[0]:.6f}"),
                }
            )

        if early_stopping(test_f1, model):
            tqdm.write(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    epoch_pbar.close()
    print("Training finished.")

    if cfg.log:
        run.log({"best val f1": best_val_f1})

        model = early_stopping.get_best_model(model)

        run.finish()

if __name__ == "__main__":
    main()
