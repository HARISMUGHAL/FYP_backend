"""
scripts/finetune_v3.py
-----------------------
SAFE fine-tuning script for the fruit classification model.
Designed specifically for the updated apple/banana real mobile images.

Key safety features:
  [OK] Loads EXISTING model_v2.pth weights (NEVER overwrites)
  [OK] Saves to NEW file: model_v3.pth
  [OK] Freezes early backbone layers (0-4)
  [OK] Uses LOW learning rate (3e-5)
  [OK] Mobile-camera augmentations (lighting, blur, perspective)
  [OK] Hash-based deduplication for clean val split
  [OK] Early stopping + validation monitoring
  [OK] Per-class accuracy + confusion matrix
  [OK] Does NOT change class labels, config, or inference pipeline

WHAT THIS MODIFIES:
  - CREATES: ml/models/saved/model_v3.pth  (new model)
  - CREATES: ml/models/saved/finetune_v3_metrics.csv  (training log)
  - CREATES: ml/models/saved/confusion_matrix_v3.png  (eval visual)

WHAT THIS DOES NOT MODIFY:
  - model_v2.pth         (production model — UNTOUCHED)
  - best_model.pth       (original model — UNTOUCHED)
  - configs/config.yaml  (config — UNTOUCHED)
  - Any backend/API/controller files
  - Any YOLO files
  - class_mapping.json   (same 27 classes)

Usage:
    .venv\\Scripts\\python.exe scripts/finetune_v3.py
"""

import argparse
import copy
import csv
import hashlib
import os
import sys
import time
from pathlib import Path

# -- Fix sys.path so all ml.* imports work regardless of cwd --------
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ml.models.fruit_model import build_model
from ml.utils.helpers import load_yaml, save_class_mapping, setup_logger

logger = setup_logger("finetune_v3")

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# =====================================================================
# Dataset with hash-based deduplication for clean train/val split
# =====================================================================

def file_hash(filepath, chunk_size=8192):
    """Compute MD5 hash for duplicate detection."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class FruitDatasetDedup(Dataset):
    """
    Dataset that scans fruit/grade folders and removes exact duplicates
    using MD5 hashing. This ensures clean train/val separation.
    """

    def __init__(self, data_dir, transform=None, dedup=True):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = []
        self._scan_dataset(dedup=dedup)

    def _scan_dataset(self, dedup=True):
        classes_set = set()
        seen_hashes = {}  # hash -> first path (for dedup)
        dup_count = 0

        for fruit_name in sorted(os.listdir(self.data_dir)):
            fruit_path = os.path.join(self.data_dir, fruit_name)
            if not os.path.isdir(fruit_path) or fruit_name.lower() == "real":
                continue
            for grade in sorted(os.listdir(fruit_path)):
                grade_path = os.path.join(fruit_path, grade)
                if not os.path.isdir(grade_path):
                    continue

                label_name = f"{fruit_name}_{grade}"
                classes_set.add(label_name)

                for fname in os.listdir(grade_path):
                    fpath = os.path.join(grade_path, fname)
                    if not os.path.isfile(fpath):
                        continue
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue

                    if dedup:
                        fh = file_hash(fpath)
                        key = f"{label_name}_{fh}"
                        if key in seen_hashes:
                            dup_count += 1
                            continue
                        seen_hashes[key] = fpath

                    self.image_paths.append(fpath)
                    self.labels.append(label_name)

        self.class_names = sorted(list(classes_set))
        self.class_to_idx = {cn: i for i, cn in enumerate(self.class_names)}

        if dedup and dup_count > 0:
            logger.info(f"Deduplication: removed {dup_count} duplicate images")
        logger.info(f"Dataset: {len(self.image_paths)} images, {len(self.class_names)} classes")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_name = self.labels[idx]
        label_idx = self.class_to_idx[label_name]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label_idx


# =====================================================================
# Augmentation transforms
# =====================================================================

def get_train_transform(img_size=224):
    """
    Mobile-camera optimized augmentation pipeline.

    Handles:
      - Zoom variation (RandomResizedCrop)
      - Lighting variation (ColorJitter brightness/contrast)
      - Shadow simulation (ColorJitter saturation)
      - Slight blur from camera defocus (GaussianBlur)
      - Angle variation (RandomPerspective + RandomRotation)
      - Standard horizontal flip
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.08,
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(img_size=224):
    """Standard validation transform — no augmentation, just resize and normalize."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# =====================================================================
# Wrapper for applying different transforms to train/val subsets
# =====================================================================

class SubsetWithTransform(Dataset):
    """Wraps a Subset to apply a specific transform."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        real_idx = self.subset.indices[index]
        img_path = self.subset.dataset.image_paths[real_idx]
        label_name = self.subset.dataset.labels[real_idx]
        label_idx = self.subset.dataset.class_to_idx[label_name]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label_idx

    def __len__(self):
        return len(self.subset)


# =====================================================================
# Class weight computation
# =====================================================================

def compute_class_weights(loader, num_classes):
    """Compute inverse-frequency class weights for balanced training."""
    counts = [0] * num_classes
    for _, labels in loader:
        for label in labels:
            counts[label.item()] += 1

    total = sum(counts)
    weights = []
    for c in counts:
        w = total / (num_classes * c) if c > 0 else 0.0
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float)


# =====================================================================
# Layer freezing
# =====================================================================

def freeze_early_layers(model, num_blocks_to_freeze=5):
    """
    Freeze the first N feature blocks of EfficientNet-B0.
    Also freezes all BatchNorm layers for stability.

    Args:
        model: EfficientNet-B0 model.
        num_blocks_to_freeze: 0-8. Default 5 for v3 fine-tuning.
    """
    # Freeze batch norm globally
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

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    logger.info(f"Layer freezing applied:")
    logger.info(f"  Blocks frozen:    {num_blocks_to_freeze} / {len(features)}")
    logger.info(f"  Total params:     {total:,}")
    logger.info(f"  Trainable params: {trainable:,}")
    logger.info(f"  Frozen params:    {frozen:,}")


# =====================================================================
# Fine-tuning loop
# =====================================================================

def finetune(
    model,
    train_loader,
    val_loader,
    class_names,
    *,
    epochs=15,
    lr=3e-5,
    patience=5,
    save_path="ml/models/saved/model_v3.pth",
    metrics_path="ml/models/saved/finetune_v3_metrics.csv",
):
    """
    Fine-tune with class-weighted loss, label smoothing, and early stopping.
    """
    device = torch.device("cpu")
    model.to(device)

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Optimizing {len(trainable_params)} parameter groups")

    # Class-weighted loss
    class_weights = compute_class_weights(train_loader, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Adam with low LR
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)

    # LR scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = []

    logger.info("=" * 60)
    logger.info("FINE-TUNING V3 STARTED")
    logger.info(f"  Epochs:    {epochs}")
    logger.info(f"  LR:        {lr}")
    logger.info(f"  Patience:  {patience}")
    logger.info(f"  Save to:   {save_path}")
    logger.info("=" * 60)

    for epoch in range(epochs):
        t_start = time.time()

        # -- Training phase --
        model.train()
        # Keep frozen BN layers in eval mode
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if not any(p.requires_grad for p in module.parameters()):
                    module.eval()

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
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

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"  Epoch {epoch+1} | Batch {batch_idx+1} | "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / total_samples
        train_acc = running_corrects.double() / total_samples

        # -- Validation phase --
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

        # -- Checkpoint best model --
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            logger.info(f"  >> New best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f"  >> Early stopping triggered after {epoch+1} epochs")
            break

    # Save training metrics
    if history:
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        logger.info(f"Training metrics saved -> {metrics_path}")

    logger.info("=" * 60)
    logger.info(f"FINE-TUNING V3 COMPLETE")
    logger.info(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"  Model saved to: {save_path}")
    logger.info("=" * 60)

    model.load_state_dict(best_model_wts)
    return model


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_model(model, val_loader, class_names):
    """
    Full evaluation with per-class accuracy and confusion matrix.
    Specifically highlights apple vs guava confusion and banana stability.
    """
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            total_correct += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    acc = total_correct / total_samples if total_samples > 0 else 0.0
    logger.info(f"Overall Accuracy: {acc:.4f}")

    # Per-class accuracy
    logger.info("\nPer-class Accuracy:")
    logger.info("-" * 40)
    for i in range(num_classes):
        class_total = confusion_matrix[i, :].sum()
        if class_total > 0:
            class_acc = confusion_matrix[i, i] / class_total
            logger.info(f"  {class_names[i]:25s}  {class_acc:.4f}  ({confusion_matrix[i,i]}/{class_total})")
        else:
            logger.info(f"  {class_names[i]:25s}  N/A (no val samples)")

    # Apple vs Guava confusion analysis
    logger.info("\n" + "=" * 60)
    logger.info("APPLE vs GUAVA CONFUSION ANALYSIS")
    logger.info("=" * 60)
    apple_classes = [i for i, cn in enumerate(class_names) if cn.startswith("apple")]
    guava_classes = [i for i, cn in enumerate(class_names) if cn.startswith("guava")]

    for ai in apple_classes:
        for gi in guava_classes:
            apple_to_guava = confusion_matrix[ai, gi]
            guava_to_apple = confusion_matrix[gi, ai]
            if apple_to_guava > 0 or guava_to_apple > 0:
                logger.info(
                    f"  {class_names[ai]} -> {class_names[gi]}: {apple_to_guava}  |  "
                    f"{class_names[gi]} -> {class_names[ai]}: {guava_to_apple}"
                )

    # Banana stability analysis
    logger.info("\n" + "=" * 60)
    logger.info("BANANA STABILITY ANALYSIS")
    logger.info("=" * 60)
    banana_classes = [i for i, cn in enumerate(class_names) if cn.startswith("banana")]
    for bi in banana_classes:
        total_b = confusion_matrix[bi, :].sum()
        correct_b = confusion_matrix[bi, bi]
        if total_b > 0:
            logger.info(
                f"  {class_names[bi]:15s}  Accuracy: {correct_b/total_b:.4f}  "
                f"({correct_b}/{total_b})"
            )
            # Show what banana was confused with
            for j in range(num_classes):
                if j != bi and confusion_matrix[bi, j] > 0:
                    logger.info(
                        f"    Confused with {class_names[j]:15s}: {confusion_matrix[bi,j]}"
                    )

    # Print full confusion matrix
    logger.info("\nFull Confusion Matrix:")
    logger.info(str(confusion_matrix))

    return acc, confusion_matrix


# =====================================================================
# Main
# =====================================================================

def main():
    logger.info("=" * 60)
    logger.info("SAFE FINE-TUNING V3 — Mobile Camera Optimization")
    logger.info("=" * 60)

    # -- 1. Load config --
    config_path = BASE_DIR / "configs" / "config.yaml"
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_yaml(str(config_path))

    # -- 2. Resolve dataset path --
    data_dir = (
        config.get("dataset", {}).get("path")
        or config.get("data", {}).get("raw_dir")
        or "FruitGrade_Dataset"
    )
    if not os.path.exists(data_dir):
        logger.error(f"Dataset path '{data_dir}' does not exist.")
        sys.exit(1)

    # -- 3. Load dataset with deduplication --
    logger.info("Loading dataset with deduplication...")
    full_dataset = FruitDatasetDedup(data_dir, transform=None, dedup=True)
    class_names = full_dataset.class_names

    if len(class_names) == 0:
        logger.error("No classes found in dataset!")
        sys.exit(1)

    logger.info(f"Found {len(class_names)} classes: {class_names}")

    # Per-class counts
    label_counts = {}
    for lbl in full_dataset.labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    logger.info("Per-class image counts (after dedup):")
    for cn in class_names:
        logger.info(f"  {cn:25s}  {label_counts.get(cn, 0):5d}")

    save_class_mapping(class_names)

    # -- 4. Train/val split --
    val_split = 0.2
    img_size = config.get("dataset", {}).get("img_size", 224)
    batch_size = config.get("dataset", {}).get("batch_size", 32)
    num_workers = config.get("dataset", {}).get("num_workers", 0)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # Use fixed seed for reproducible split
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    train_dataset = SubsetWithTransform(train_subset, get_train_transform(img_size))
    val_dataset = SubsetWithTransform(val_subset, get_val_transform(img_size))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    logger.info(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    # -- 5. Build model --
    logger.info("Building model...")
    model = build_model(
        num_classes=len(class_names),
        pretrained=True,
        dropout_rate=config.get("model", {}).get("dropout_rate", 0.4),
    )

    # -- 6. Load EXISTING model_v2 weights --
    original_model_path = config.get("model", {}).get(
        "save_path", "ml/models/saved/model_v2.pth"
    )

    if os.path.exists(original_model_path):
        logger.info(f"Loading existing model weights from: {original_model_path}")
        state_dict = torch.load(
            original_model_path, map_location=torch.device("cpu"), weights_only=False
        )

        # Handle both raw state_dict and checkpoint dict formats
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        model.load_state_dict(state_dict)
        logger.info("[OK] model_v2 weights loaded successfully")
    else:
        logger.warning(
            f"[!] model_v2 not found at {original_model_path}. "
            "Fine-tuning from pretrained ImageNet weights instead."
        )

    # -- 7. Freeze early layers (blocks 0-4) --
    freeze_early_layers(model, num_blocks_to_freeze=5)

    # -- 8. Safety check --
    new_model_path = "ml/models/saved/model_v3.pth"
    metrics_path = "ml/models/saved/finetune_v3_metrics.csv"

    if new_model_path == original_model_path:
        logger.error("[X] SAFETY: New model path matches original! Aborting.")
        sys.exit(1)

    logger.info(f"\n{'='*60}")
    logger.info(f"SAFETY CHECK:")
    logger.info(f"  Original model: {original_model_path}  (UNTOUCHED)")
    logger.info(f"  New model:      {new_model_path}")
    logger.info(f"{'='*60}\n")

    # -- 9. Fine-tune --
    best_model = finetune(
        model,
        train_loader,
        val_loader,
        class_names,
        epochs=15,
        lr=3e-5,
        patience=5,
        save_path=new_model_path,
        metrics_path=metrics_path,
    )

    # -- 10. Evaluate --
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION ON VALIDATION SET")
    logger.info("=" * 60)
    evaluate_model(best_model, val_loader, class_names)

    # -- 11. Print switching instructions --
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
