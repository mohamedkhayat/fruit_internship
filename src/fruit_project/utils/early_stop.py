# SPDX-FileCopyrightText: 2025 Mohamed Khayat
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
import pathlib
from tqdm import tqdm
import wandb
from wandb.sdk.wandb_run import Run
import torch.nn as nn
from typing import List, Optional, Tuple
from pathlib import Path


class EarlyStopping:
    def __init__(
        self,
        patience: int,
        delta: float,
        path: str,
        name: str,
        run: Run,
        log: bool = False,
        upload: bool = False,
    ):
        """
        Initializes the EarlyStopping object.

        Args:
            patience (int): Number of epochs to wait before stopping if no improvement.
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
            path (str): Directory path to save model checkpoints.
            name (str): Name prefix for saved model files.
            cfg (DictConfig): Configuration object.
            run (Run): WandB run object for logging artifacts.
        """
        self.patience = patience
        self.delta = delta

        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.run = run
        self.best_metric: Optional[float] = None
        self.counter = 0
        self.earlystop = False

        self.log = log
        self.upload = upload

        self.saved_checkpoints: List[Tuple[float, Path]] = []

    def __call__(self, val_metric: float, model: nn.Module) -> bool:
        """
        Checks if early stopping criteria are met and saves the model if the metric improves.

        Args:
            val_metric (float): Validation metric to monitor.
            model (nn.Module): PyTorch model to save.

        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """
        if self.best_metric is None:
            self.best_metric = val_metric
            tqdm.write("saved model weights")
            self.save_model(model, val_metric)

        elif val_metric <= self.best_metric + self.delta:
            self.counter += 1
        else:
            self.best_metric = val_metric
            self.save_model(model, val_metric)
            self.counter = 0
            tqdm.write("saved model weights")

        if self.counter >= self.patience:
            self.earlystop = True

        return self.earlystop

    def save_model(self, model: nn.Module, val_metric: float):
        """
        Saves the model checkpoint.

        Args:
            model (nn.Module): PyTorch model to save.
            val_metric (float): Validation metric value used for naming the checkpoint file.

        Returns:
            None
        """
        filename = f"{self.name}_{val_metric:.4f}.pth".replace("=", "-")
        full_path = self.path / filename
        torch.save(model.state_dict(), full_path)
        self.saved_checkpoints.append((val_metric, full_path))

    def cleanup_checkpoints(self):
        """
        Deletes all saved checkpoints except the best one.

        Returns:
            None
        """
        if not self.saved_checkpoints:
            tqdm.write("No checkpoints to clean up.")
            return

        tqdm.write("cleaning up old checkpoints...")
        _, best_path = max(self.saved_checkpoints, key=lambda x: x[0])

        for _, path in self.saved_checkpoints:
            if path != best_path and path.exists():
                try:
                    path.unlink()
                    tqdm.write(f"deleted {path.name}")
                except Exception as e:
                    tqdm.write(f"could not delete {path.name}: {e}")

        tqdm.write(f"kept best model: {best_path.name}")

    def get_best_model(self, model: nn.Module) -> nn.Module:
        """
        Loads the best model checkpoint and sets the model to evaluation mode.

        Args:
            model (nn.Module): PyTorch model to load the best checkpoint into.

        Returns:
            nn.Module: The model with the best checkpoint loaded.
        """
        self.cleanup_checkpoints()
        tqdm.write("loading best model")
        model.eval()

        if len(self.saved_checkpoints) > 0:
            _, best_path = max(self.saved_checkpoints, key=lambda x: x[0])
            model.load_state_dict(torch.load(best_path, weights_only=True))
            if self.log and self.upload:
                artifact = wandb.Artifact(
                    name=f"{self.name.split('_')[0]}",
                    type="model-earlystopping-bestmodel",
                    description="best model at epoch",
                )
                artifact.add_file(best_path)
                self.run.log_artifact(artifact)
                artifact.wait()
        return model
