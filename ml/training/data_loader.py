"""
ml/training/data_loader.py
--------------------------
Builds torchvision DataLoaders for train / val / test splits.

v2 — Stronger augmentation pipeline:
  - RandomResizedCrop (scale invariance, partial-fruit views)
  - Stronger ColorJitter (brightness/contrast/saturation/hue)
  - RandomGrayscale (prevents pure-colour bias between similar fruits)
  - Deterministic Resize(256) → CenterCrop(224) for val/test
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml.utils.helpers import setup_logger

logger = setup_logger(__name__)


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────

def get_transforms(split: str, cfg: dict) -> transforms.Compose:
    """
    Return the appropriate torchvision transform pipeline for a split.

    Training pipeline (v2):
        RandomResizedCrop  → scale-invariant fruit learning
        RandomHorizontalFlip
        RandomRotation
        ColorJitter        → stronger than v1
        RandomGrayscale    → prevents colour-only discrimination
        ToTensor → Normalize

    Val / test pipeline (deterministic):
        Resize(resize_size) → CenterCrop(center_crop) → ToTensor → Normalize

    Args:
        split: "train" | "val" | "test"
        cfg:   Config dict.

    Returns:
        torchvision.transforms.Compose
    """
    pp          = cfg["preprocessing"]
    size        = pp["image_size"]          # 224 — final spatial size
    resize_size = pp.get("resize_size", 256)  # 256 — pre-crop resize for val/test
    crop        = pp["center_crop"]         # 224
    mean        = pp["mean"]
    std         = pp["std"]

    normalise = transforms.Normalize(mean=mean, std=std)

    # ── Training: aggressive augmentation ────────────────────────────────────
    if split == "train":
        aug_list: list = []

        # 1. RandomResizedCrop — THE most important augmentation for fruits.
        #    Teaches the model to recognise fruit at any scale and position,
        #    matching real-world camera distances far better than plain Resize.
        rrc_cfg = pp.get("random_resized_crop", {})
        if rrc_cfg.get("enabled", True):
            scale_min = float(rrc_cfg.get("scale_min", 0.60))
            scale_max = float(rrc_cfg.get("scale_max", 1.00))
            aug_list.append(
                transforms.RandomResizedCrop(
                    size,
                    scale=(scale_min, scale_max),
                    ratio=(0.75, 1.333),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            )
        else:
            # Fallback if RRC is disabled in config
            aug_list.append(transforms.Resize((size, size)))

        # 2. Horizontal flip
        if pp.get("random_horizontal_flip", True):
            aug_list.append(transforms.RandomHorizontalFlip())

        # 3. Rotation
        rot_deg = pp.get("random_rotation_deg", 0)
        if rot_deg > 0:
            aug_list.append(transforms.RandomRotation(rot_deg))

        # 4. ColorJitter — stronger than v1 to teach colour-invariant features
        cj = pp.get("color_jitter", {})
        if cj:
            aug_list.append(
                transforms.ColorJitter(
                    brightness=cj.get("brightness", 0),
                    contrast=cj.get("contrast", 0),
                    saturation=cj.get("saturation", 0),
                    hue=cj.get("hue", 0),
                )
            )

        # 5. RandomGrayscale — small probability forces shape-based learning,
        #    not just colour-based.  Helps distinguish apple vs orange vs apricot.
        gs_prob = pp.get("random_grayscale_prob", 0.0)
        if gs_prob > 0:
            aug_list.append(transforms.RandomGrayscale(p=gs_prob))

        aug_list += [transforms.ToTensor(), normalise]
        return transforms.Compose(aug_list)

    # ── Val / test: deterministic ─────────────────────────────────────────────
    # Resize to slightly larger than crop, then center-crop.
    # This is the standard torchvision EfficientNet evaluation protocol and
    # gives slightly better accuracy than a plain Resize(224).
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        normalise,
    ])


# ─────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────

def get_dataloaders(cfg: dict) -> dict[str, DataLoader]:
    """
    Build and return DataLoaders for all three splits.

    Expects directories:
        cfg['data']['train_dir']/<class_label>/
        cfg['data']['val_dir']/<class_label>/
        cfg['data']['test_dir']/<class_label>/

    Args:
        cfg: Config dict.

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    batch_size  = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    pin_memory  = cfg["training"]["pin_memory"]

    split_dirs = {
        "train": cfg["data"]["train_dir"],
        "val":   cfg["data"]["val_dir"],
        "test":  cfg["data"]["test_dir"],
    }

    loaders: dict[str, DataLoader] = {}

    for split, dir_path in split_dirs.items():
        root = Path(dir_path)
        if not root.exists() or not any(root.iterdir()):
            logger.warning(
                f"  Split directory '{root}' is missing or empty. "
                f"Run dataset_prep.prepare_dataset() first."
            )
            loaders[split] = None  # type: ignore[assignment]
            continue

        tfm     = get_transforms(split, cfg)
        dataset = datasets.ImageFolder(root=str(root), transform=tfm)

        shuffle = split == "train"
        loader  = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            # persistent_workers requires num_workers > 0
            persistent_workers=(num_workers > 0),
        )

        logger.info(
            f"  [{split}] {len(dataset)} images | "
            f"{len(loader)} batches | "
            f"{len(dataset.classes)} classes"
        )
        loaders[split] = loader

    # Expose class info through the train loader's dataset
    if loaders.get("train") is not None:
        classes = loaders["train"].dataset.classes  # type: ignore[union-attr]
        logger.info(f"Classes ({len(classes)}): {classes}")

    return loaders


def get_class_names(cfg: dict) -> list[str]:
    """
    Return the ordered list of class names from the train split directory.
    This matches the order ImageFolder assigns integer labels.
    """
    train_root = Path(cfg["data"]["train_dir"])
    if not train_root.exists():
        raise FileNotFoundError(
            f"Train directory not found: {train_root}. "
            "Run dataset_prep.prepare_dataset() first."
        )
    dataset = datasets.ImageFolder(root=str(train_root))
    return dataset.classes
