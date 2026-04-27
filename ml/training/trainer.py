"""
ml/training/trainer.py
-----------------------
Training and validation loops for EfficientNet-B0.

Features:
  - Per-epoch train / val pass with loss & accuracy tracking
  - Best-model checkpoint based on validation accuracy
  - CosineAnnealingLR scheduler
  - Early stopping
  - Metrics CSV export
"""

import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.models.model_builder import save_checkpoint
from ml.utils.helpers import ensure_dir, setup_logger

logger = setup_logger(__name__)


# ─────────────────────────────────────────────
# Single-epoch passes
# ─────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one full training epoch.

    Returns:
        (avg_loss, accuracy_percentage)
    """
    model.train()
    running_loss   = 0.0
    correct        = 0
    total          = 0

    pbar = tqdm(loader, desc="  [train]", leave=False, unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds  = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one full validation / test pass (no gradient computation).

    Returns:
        (avg_loss, accuracy_percentage)
    """
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    pbar = tqdm(loader, desc="  [val]  ", leave=False, unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds  = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ─────────────────────────────────────────────
# Full training loop
# ─────────────────────────────────────────────

def train(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    cfg: dict,
    device: torch.device,
) -> nn.Module:
    """
    Full training loop with scheduler, early stopping and checkpointing.

    Args:
        model:       Model to train (already on device).
        dataloaders: {"train": DataLoader, "val": DataLoader, ...}
        cfg:         Config dict.
        device:      Training device.

    Returns:
        Model loaded with the best checkpoint weights.
    """
    t_cfg = cfg["training"]
    epochs        = t_cfg["epochs"]
    patience      = t_cfg["early_stopping_patience"]
    metrics_csv   = Path(t_cfg["metrics_csv"])
    ensure_dir(metrics_csv.parent)

    # ── Optimiser ─────────────────────────────────────────────────
    opt_name = t_cfg["optimizer"].lower()
    lr       = t_cfg["learning_rate"]
    wd       = t_cfg["weight_decay"]

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=wd
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # ── Scheduler ─────────────────────────────────────────────────
    scheduler_name = t_cfg.get("scheduler", "none").lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_cfg.get("t_max", epochs)
        )
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None

    # ── Loss ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # ── Tracking ──────────────────────────────────────────────────
    best_val_acc   = 0.0
    no_improve     = 0
    best_ckpt_path = Path(cfg["model"]["save_dir"]) / cfg["model"]["checkpoint_name"]

    # CSV header
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    logger.info("=" * 55)
    logger.info(f"Training on {device} | epochs={epochs} | lr={lr} | optimizer={opt_name}")
    logger.info("=" * 55)

    for epoch in range(1, epochs + 1):
        t_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Train pass ────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], optimizer, criterion, device
        )

        # ── Val pass ──────────────────────────────────────────────
        val_loss, val_acc = validate(
            model, dataloaders["val"], criterion, device
        )

        # ── Scheduler step ────────────────────────────────────────
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t_start
        logger.info(
            f"Epoch [{epoch:>3}/{epochs}] "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%  "
            f"lr={current_lr:.2e}  ({elapsed:.1f}s)"
        )

        # ── CSV log ───────────────────────────────────────────────
        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}", f"{train_acc:.4f}",
                f"{val_loss:.6f}",   f"{val_acc:.4f}",
                f"{current_lr:.8f}",
            ])

        # ── Best checkpoint ───────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            save_checkpoint(
                model, optimizer, epoch,
                {"val_acc": round(val_acc, 4), "val_loss": round(val_loss, 6)},
                cfg,
                filename=cfg["model"]["checkpoint_name"],
            )
            logger.info(f"  ★ New best val_acc={val_acc:.2f}% — checkpoint saved")
        else:
            no_improve += 1
            logger.info(f"  No improvement ({no_improve}/{patience})")

        # ── Early stopping ────────────────────────────────────────
        if no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break

    # ── Save final model ──────────────────────────────────────────
    save_checkpoint(
        model, optimizer, epoch,
        {"val_acc": round(val_acc, 4)},
        cfg,
        filename=cfg["model"]["final_name"],
    )

    # ── Reload best weights ───────────────────────────────────────
    logger.info(f"Loading best checkpoint from {best_ckpt_path}")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    logger.info(f"Training complete. Best val_acc = {best_val_acc:.2f}%")
    return model
