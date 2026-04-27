"""
ml/training/data_loader.py
--------------------------
Builds torchvision DataLoaders for train / val / test splits.
Applies data augmentation on the training set; deterministic
resize + centre-crop on val / test.
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

    Args:
        split: "train" | "val" | "test"
        cfg:   Config dict.

    Returns:
        torchvision.transforms.Compose
    """
    pp = cfg["preprocessing"]
    size   = pp["image_size"]
    crop   = pp["center_crop"]
    mean   = pp["mean"]
    std    = pp["std"]

    # ── Normalise ──────────────────────────────────────────────────
    normalise = transforms.Normalize(mean=mean, std=std)

    if split == "train":
        aug_list: list = [
            transforms.Resize((size, size)),
        ]
        if pp.get("random_horizontal_flip"):
            aug_list.append(transforms.RandomHorizontalFlip())

        rot_deg = pp.get("random_rotation_deg", 0)
        if rot_deg > 0:
            aug_list.append(transforms.RandomRotation(rot_deg))

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

        aug_list += [transforms.ToTensor(), normalise]
        return transforms.Compose(aug_list)

    # val / test — deterministic
    return transforms.Compose([
        transforms.Resize((size, size)),
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

        tfm = get_transforms(split, cfg)
        dataset = datasets.ImageFolder(root=str(root), transform=tfm)

        shuffle = split == "train"
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
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
