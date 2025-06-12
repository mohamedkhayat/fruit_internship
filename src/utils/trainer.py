from torchmetrics import F1Score
import torch
import torch.nn as nn
from torch.amp import GradScaler
from tqdm import tqdm
from .general import is_hf_model


def make_forward_step(model: nn.Module):
    if is_hf_model(model):
        return lambda x: model(x).logits
    else:
        return lambda x: model(x)


def train(
    model: nn.Module,
    device: torch.device,
    train_dl,
    criterion: nn.Module,
    scaler: GradScaler,
    num_classes,
    optimizer,
    current_epoch,
):
    forward_step = make_forward_step(model)
    model.train()
    loss = 0.0

    f1 = F1Score(
        task="multiclass",
        num_classes=num_classes,
        average="weighted",
        compute_on_cpu=True,
    )

    device_str = str(device).split(":")[0]
    progress_bar = tqdm(
        train_dl,
        desc=f"Epoch {current_epoch} Training",
        leave=True,
        bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
    )

    for batch_idx, (x, y) in enumerate(progress_bar):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device_str, dtype=torch.float16):
            # out = model(x).logits
            out = forward_step(x)
            batch_loss = criterion(out, y)

        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss += batch_loss.item()
        f1.update(out.cpu(), y.cpu())

        current_avg_loss = loss / (batch_idx + 1)
        progress_bar.set_postfix(
            {
                "Loss": f"{current_avg_loss:.4f}",
                "Batch": f"{batch_idx + 1}/{len(train_dl)}",
            }
        )

    f1_score = f1.compute()
    f1.reset()
    loss /= len(train_dl)
    return loss, f1_score.item()


@torch.no_grad()
def eval(model: nn.Module, device, test_dl, criterion, num_classes, current_epoch):
    forward_step = make_forward_step(model)
    model.eval()

    loss = 0.0
    y_true = []
    y_pred = []

    f1 = F1Score(
        task="multiclass",
        num_classes=num_classes,
        average="weighted",
        compute_on_cpu=True,
    )

    device_str = str(device).split(":")[0]

    progress_bar = tqdm(
        test_dl,
        desc=f"Epoch {current_epoch} Evaluating",
        leave=True,
        bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
    )

    for batch_idx, (x, y) in enumerate(progress_bar):
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device_str, dtype=torch.float16):
            # out = model(x).logits
            out = forward_step(x)
            batch_loss = criterion(out, y)

        loss += batch_loss.item()
        f1.update(out.cpu(), y.cpu())

        current_avg_loss = loss / (batch_idx + 1)
        progress_bar.set_postfix(
            {
                "Loss": f"{current_avg_loss:.4f}",
                "Batch": f"{batch_idx + 1}/{len(test_dl)}",
            }
        )
        y_true.extend(y.cpu().numpy())
        y_pred.extend(torch.argmax(out, dim = 1).cpu().numpy())

    f1_score = f1.compute()
    f1.reset()
    loss /= len(test_dl)
    return loss, f1_score.item(), y_true, y_pred


def get_optimizer(
    model, head_lr=1e-5, backbone_lr=1e-3, weight_decay=0.001, freeze=False
):
    if freeze:
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if (
                    name.startswith("classifier")
                    or name == "fc.weight"
                    or name == "fc.bias"
                ):
                    head_params.append(param)
                else:
                    backbone_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": head_lr},
            ]
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=head_lr,
            weight_decay=weight_decay,
        )

    return optimizer
