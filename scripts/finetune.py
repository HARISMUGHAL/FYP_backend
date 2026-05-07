"""
scripts/finetune.py
-------------------
SAFE fine-tuning script for the fruit classification model.

Key safety features:
  ✓ Loads EXISTING trained model (best_model.pth)
  ✓ Saves to NEW versioned file (model_v2.pth) — NEVER overwrites original
  ✓ Freezes early layers — only trains classifier + last conv block
  ✓ Uses LOW learning rate (1e-4 or lower)
  ✓ Includes early stopping + validation monitoring
  ✓ Evaluates per-class accuracy + confusion matrix
  ✓ Does NOT change class labels or inference pipeline

Usage:
    python scripts/finetune.py
    python scripts/finetune.py --config configs/config.yaml --version v2
    python scripts/finetune.py --epochs 15 --lr 0.00005

IMPORTANT:
    This script does NOT modify or overwrite the original model.
    The original model remains at: ml/models/saved/best_model.pth
    The new model is saved to:     ml/models/saved/model_v2.pth
"""

import argparse
import copy
import csv
import os
import sys
import time
from pathlib import Path

# ── Fix sys.path so all ml.* imports work regardless of cwd ──────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ml.preprocessing.dataset import get_dataloaders
from ml.models.fruit_model import build_model
from ml.training.trainer import compute_class_weights
from ml.evaluation.evaluator import evaluate_model
from ml.utils.helpers import load_yaml, save_class_mapping, setup_logger

logger = setup_logger("finetune")


# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safe fine-tuning of the fruit classification model"
    )
    parser.add_argument(
        "--config",
        default=str(BASE_DIR / "configs" / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--version",
        default="v2",
        help="Version suffix for new model (default: v2 → model_v2.pth)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of fine-tuning epochs (default: 15)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00005,
        help="Learning rate (default: 5e-5, MUST be low for fine-tuning)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)",
    )
    parser.add_argument(
        "--freeze-layers",
        type=int,
        default=6,
        help="Number of EfficientNet feature blocks to freeze (0-8, default: 6)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Layer freezing
# ─────────────────────────────────────────────

def freeze_early_layers(model: nn.Module, num_blocks_to_freeze: int = 6) -> None:
    """
    Freeze the first N feature blocks of EfficientNet-B0.
    EfficientNet-B0 has 8 feature blocks (indices 0-7) + classifier.

    This prevents the early feature extractors from being modified,
    preserving what the model already learned while allowing the deeper
    layers to adapt to new data.

    Args:
        model: EfficientNet-B0 model.
        num_blocks_to_freeze: How many blocks to freeze (0=none, 8=all features).
    """
    # Freeze batch norm layers globally for stability
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    # Freeze early feature blocks
    features = model.features
    num_blocks_to_freeze = min(num_blocks_to_freeze, len(features))

    for i in range(num_blocks_to_freeze):
        for param in features[i].parameters():
            param.requires_grad = False

    # Count trainable params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    logger.info(f"Layer freezing applied:")
    logger.info(f"  Blocks frozen:    {num_blocks_to_freeze} / {len(features)}")
    logger.info(f"  Total params:     {total:,}")
    logger.info(f"  Trainable params: {trainable:,}")
    logger.info(f"  Frozen params:    {frozen:,}")


# ─────────────────────────────────────────────
# Fine-tuning loop
# ─────────────────────────────────────────────

def finetune(
    model: nn.Module,
    train_loader,
    val_loader,
    class_names: list[str],
    *,
    epochs: int = 15,
    lr: float = 5e-5,
    patience: int = 5,
    save_path: str = "ml/models/saved/model_v2.pth",
    metrics_path: str = "ml/models/saved/finetune_metrics.csv",
) -> nn.Module:
    """
    Fine-tune the model with class-weighted loss and early stopping.

    Args:
        model: Pre-loaded model with frozen early layers.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        class_names: List of class names.
        epochs: Max epochs.
        lr: Learning rate (should be LOW for fine-tuning).
        patience: Early stopping patience.
        save_path: Where to save the new model.
        metrics_path: CSV file for training metrics.

    Returns:
        Fine-tuned model with best weights loaded.
    """
    device = torch.device("cpu")
    model.to(device)

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Optimizing {len(trainable_params)} parameter groups")

    # Class-weighted loss to handle any class imbalance
    class_weights = compute_class_weights(train_loader, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Adam with low LR for fine-tuning
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)

    # Learning rate scheduler — reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = []

    logger.info("=" * 60)
    logger.info("FINE-TUNING STARTED")
    logger.info(f"  Epochs:    {epochs}")
    logger.info(f"  LR:        {lr}")
    logger.info(f"  Patience:  {patience}")
    logger.info(f"  Save to:   {save_path}")
    logger.info("=" * 60)

    for epoch in range(epochs):
        t_start = time.time()

        # ── Training phase ────────────────────────────────────
        model.train()
        # Keep frozen BN layers in eval mode
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if not any(p.requires_grad for p in module.parameters()):
                    module.eval()

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        train_loss = running_loss / total_samples
        train_acc = running_corrects.double() / total_samples

        # ── Validation phase ──────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_samples += inputs.size(0)

        val_loss = val_loss / val_samples
        val_acc = val_corrects.double() / val_samples
        elapsed = time.time() - t_start

        # Update LR scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{float(train_acc):.4f}",
            "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{float(val_acc):.4f}",
            "lr": f"{current_lr:.6f}",
        })

        # ── Checkpoint best model ─────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            logger.info(f"  ✓ New best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f"  ✗ Early stopping triggered after {epoch+1} epochs")
            break

    # ── Save training metrics ─────────────────────────────
    if history:
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        logger.info(f"Training metrics saved → {metrics_path}")

    logger.info("=" * 60)
    logger.info(f"FINE-TUNING COMPLETE")
    logger.info(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"  Model saved to: {save_path}")
    logger.info("=" * 60)

    model.load_state_dict(best_model_wts)
    return model


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # -- 1. Load config ─────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_yaml(str(config_path))

    # -- 2. Resolve dataset path ────────────────────────────────────────────────
    data_dir = (
        config.get("dataset", {}).get("path")
        or config.get("data", {}).get("raw_dir")
        or "FruitGrade_Dataset"
    )
    if not os.path.exists(data_dir):
        logger.error(f"Dataset path '{data_dir}' does not exist.")
        sys.exit(1)

    # -- 3. Load dataset ───────────────────────────────────────────────────────
    logger.info("Loading dataset...")
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        img_size=config.get("dataset", {}).get("img_size", 224),
        batch_size=config.get("dataset", {}).get("batch_size", 32),
        val_split=config.get("dataset", {}).get("val_split", 0.2),
        num_workers=config.get("dataset", {}).get("num_workers", 0),
    )

    if len(class_names) == 0:
        logger.error("No classes found in dataset!")
        sys.exit(1)

    logger.info(f"Found {len(class_names)} classes: {class_names}")
    save_class_mapping(class_names)

    # -- 4. Build model with same architecture ──────────────────────────────────
    logger.info("Building model...")
    model = build_model(
        num_classes=len(class_names),
        pretrained=True,
        dropout_rate=config.get("model", {}).get("dropout_rate", 0.4),
    )

    # -- 5. Load EXISTING trained weights ───────────────────────────────────────
    original_model_path = config.get("model", {}).get("save_path", "ml/models/saved/best_model.pth")

    if os.path.exists(original_model_path):
        logger.info(f"Loading existing model weights from: {original_model_path}")
        state_dict = torch.load(original_model_path, map_location=torch.device("cpu"), weights_only=False)

        # Handle both raw state_dict and checkpoint dict formats
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        model.load_state_dict(state_dict)
        logger.info("✓ Original model weights loaded successfully")
    else:
        logger.warning(
            f"⚠ Original model not found at {original_model_path}. "
            "Fine-tuning from pretrained ImageNet weights instead."
        )

    # -- 6. Freeze early layers ─────────────────────────────────────────────────
    freeze_early_layers(model, num_blocks_to_freeze=args.freeze_layers)

    # -- 7. Set save paths (NEVER overwrite original) ──────────────────────────
    version = args.version
    new_model_path = f"ml/models/saved/model_{version}.pth"
    metrics_path = f"ml/models/saved/finetune_{version}_metrics.csv"

    if new_model_path == original_model_path:
        logger.error("❌ SAFETY: New model path matches original! Aborting.")
        logger.error("   Use --version to specify a different version suffix.")
        sys.exit(1)

    logger.info(f"\n{'='*60}")
    logger.info(f"SAFETY CHECK:")
    logger.info(f"  Original model: {original_model_path}  (UNTOUCHED)")
    logger.info(f"  New model:      {new_model_path}")
    logger.info(f"{'='*60}\n")

    # -- 8. Fine-tune ──────────────────────────────────────────────────────────
    best_model = finetune(
        model,
        train_loader,
        val_loader,
        class_names,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        save_path=new_model_path,
        metrics_path=metrics_path,
    )

    # -- 9. Evaluate ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION ON VALIDATION SET")
    logger.info("=" * 60)
    evaluate_model(best_model, val_loader, class_names)

    # -- 10. Print switching instructions ──────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("HOW TO SWITCH TO THE NEW MODEL")
    logger.info("=" * 60)
    logger.info(f"  1. Open: configs/config.yaml")
    logger.info(f"  2. Change model.save_path to: {new_model_path}")
    logger.info(f"  3. Restart the backend server")
    logger.info(f"")
    logger.info(f"  To REVERT: change model.save_path back to: {original_model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
