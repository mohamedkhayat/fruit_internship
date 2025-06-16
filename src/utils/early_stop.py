import torch
import pathlib
from tqdm import tqdm
import wandb


class EarlyStopping:
    def __init__(self, patience, delta, path, name, cfg):
        self.patience = patience
        self.delta = delta

        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.cfg = cfg
        self.best_metric = None
        self.counter = 0
        self.earlystop = False

        self.saved_checkpoints = []

    def __call__(self, val_metric, model):
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
            tqdm.write(f"saved model weights")

        if self.counter >= self.patience:
            print("")
            tqdm.write(f"early stop triggered")
            self.earlystop = True

        return self.earlystop

    def save_model(self, model, val_metric):
        filename = f"{self.name}_{val_metric:.4f}.pth"
        full_path = self.path / filename
        torch.save(model.state_dict(), full_path)
        self.saved_checkpoints.append((val_metric, full_path))

    def cleanup_checkpoints(self):
        if not self.saved_checkpoints:
            tqdm.write(f"No checkpoints to clean up.")
            return

        tqdm.write(f"cleaning up old checkpoints...")
        best_val, best_path = max(self.saved_checkpoints, key=lambda x: x[0])

        for val, path in self.saved_checkpoints:
            if path != best_path and path.exists():
                try:
                    path.unlink()
                    tqdm.write(f"deleted {path.name}")
                except Exception as e:
                    tqdm.write(f"could not delete {path.name}: {e}")

        tqdm.write(f"kept best model: {best_path.name}")

    def get_best_model(self, model):
        self.cleanup_checkpoints()
        tqdm.write(f"loading best model")
        model.eval()

        if len(self.saved_checkpoints) > 0:
            _, best_path = max(self.saved_checkpoints, key=lambda x: x[0])
            model.load_state_dict(torch.load(best_path, weights_only=True))
            artifact = wandb.Artifact(
                name=f"{self.cfg.model.name}",
                type="model-earlystopping-bestmodel",
                description=f"best model at epoch",
            )
            artifact.add_file(best_path)
            self.run.log_artifact(artifact)
            artifact.wait()
        return model
