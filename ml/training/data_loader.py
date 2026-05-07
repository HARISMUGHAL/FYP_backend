"""
ml/training/data_loader.py
--------------------------
Builds torchvision DataLoaders for train / val / test splits.

v3 — Flexible config access:
  Supports BOTH legacy cfg["data"]["train_dir"] AND new cfg["dataset"]["path"]
  format.  Falls back gracefully so the same code works regardless of
  which config schema is present.

Augmentation pipeline:
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
# Config access helpers
# ─────────────────────────────────────────────

def _resolve_dataset_path(cfg: dict) -> str:
    """
    Return the base dataset path from config, supporting both formats:
      - cfg["dataset"]["path"]          (NEW)
      - cfg["data"]["raw_dir"]          (LEGACY)
      - Fallback: "FruitGrade_Dataset"
    """
    dataset_cfg = cfg.get("dataset", {})
    data_cfg = cfg.get("data", {})

    path = dataset_cfg.get("path") or data_cfg.get("raw_dir") or "FruitGrade_Dataset"
    return path


def _resolve_split_dirs(cfg: dict) -> dict[str, str]:
    """
    Return paths for train / val / test splits.

    Priority:
      1. cfg["data"]["train_dir"] etc.  (explicit split dirs)
      2. cfg["dataset"]["path"]         (single root — used for all splits)
      3. Fallback "FruitGrade_Dataset"
    """
    data_cfg = cfg.get("data", {})
    fallback = _resolve_dataset_path(cfg)

    return {
        "train": data_cfg.get("train_dir", fallback),
        "val":   data_cfg.get("val_dir",   fallback),
        "test":  data_cfg.get("test_dir",  fallback),
    }


def _get_preprocessing(cfg: dict) -> dict:
    """
    Return preprocessing params with safe defaults.
    Checks cfg["preprocessing"] first, then falls back to sensible defaults.
    """
    pp = cfg.get("preprocessing", {})
    return {
        "image_size":   pp.get("image_size", cfg.get("dataset", {}).get("img_size", 224)),
        "resize_size":  pp.get("resize_size", 256),
        "center_crop":  pp.get("center_crop", 224),
        "mean":         pp.get("mean", [0.485, 0.456, 0.406]),
        "std":          pp.get("std",  [0.229, 0.224, 0.225]),
    }


def _get_training_params(cfg: dict) -> dict:
    """
    Return training params with safe defaults.
    Merges cfg["training"] with cfg["dataset"] fallbacks.
    """
    t = cfg.get("training", {})
    d = cfg.get("dataset", {})
    return {
        "batch_size":   t.get("batch_size",   d.get("batch_size", 32)),
        "num_workers":  t.get("num_workers",  d.get("num_workers", 0)),
        "pin_memory":   t.get("pin_memory",   False),
    }


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────

def get_transforms(split: str, cfg: dict) -> transforms.Compose:
    """
    Return the appropriate torchvision transform pipeline for a split.

    Training pipeline:
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
    pp_raw      = cfg.get("preprocessing", {})
    pp          = _get_preprocessing(cfg)
    size        = pp["image_size"]
    resize_size = pp["resize_size"]
    crop        = pp["center_crop"]
    mean        = pp["mean"]
    std         = pp["std"]

    normalise = transforms.Normalize(mean=mean, std=std)

    # ── Training: aggressive augmentation ────────────────────────────────────
    if split == "train":
        aug_list: list = []

        # 1. RandomResizedCrop — THE most important augmentation for fruits.
        rrc_cfg = pp_raw.get("random_resized_crop", {})
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
        if pp_raw.get("random_horizontal_flip", True):
            aug_list.append(transforms.RandomHorizontalFlip())

        # 3. Rotation
        rot_deg = pp_raw.get("random_rotation_deg", 0)
        if rot_deg > 0:
            aug_list.append(transforms.RandomRotation(rot_deg))

        # 4. ColorJitter — stronger than v1 to teach colour-invariant features
        cj = pp_raw.get("color_jitter", {})
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
        gs_prob = pp_raw.get("random_grayscale_prob", 0.0)
        if gs_prob > 0:
            aug_list.append(transforms.RandomGrayscale(p=gs_prob))

        aug_list += [transforms.ToTensor(), normalise]
        return transforms.Compose(aug_list)

    # ── Val / test: deterministic ─────────────────────────────────────────────
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

    Supports BOTH config formats:
      - cfg["data"]["train_dir"] / cfg["data"]["val_dir"] / cfg["data"]["test_dir"]
      - cfg["dataset"]["path"]  (single path used for all splits)

    Args:
        cfg: Config dict.

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    tp          = _get_training_params(cfg)
    batch_size  = tp["batch_size"]
    num_workers = tp["num_workers"]
    pin_memory  = tp["pin_memory"]

    split_dirs = _resolve_split_dirs(cfg)

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

    Supports both config formats via _resolve_split_dirs().
    """
    split_dirs = _resolve_split_dirs(cfg)
    train_root = Path(split_dirs["train"])

    if not train_root.exists():
        raise FileNotFoundError(
            f"Train directory not found: {train_root}. "
            "Run dataset_prep.prepare_dataset() first."
        )
    dataset = datasets.ImageFolder(root=str(train_root))
    logger.info(f"Class names resolved from: {train_root} → {dataset.classes}")
    return dataset.classes
