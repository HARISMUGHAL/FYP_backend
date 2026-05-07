"""
ml/preprocessing/dataset_prep.py
---------------------------------
Scans FruitGrade_Dataset/, cleans corrupt images, then copies images
into ml/data/{train,val,test}/<fruit>_<grade>/ with a 70/20/10 split.

Usage (standalone):
    python -m ml.preprocessing.dataset_prep --config configs/config.yaml
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

from ml.utils.helpers import ensure_dir, load_yaml, set_seed, setup_logger

logger = setup_logger(__name__)

# Grades we recognise (case-normalised for comparison)
VALID_GRADES = {"a", "b", "c"}


# ─────────────────────────────────────────────
# 1. Scan raw dataset
# ─────────────────────────────────────────────

def scan_raw_dataset(cfg: dict) -> dict[str, list[Path]]:
    """
    Walk raw_dir/<fruit>/<grade>/ and collect all image paths grouped
    by class label "<fruit>_<grade>".

    Skips folders listed in cfg['data']['skip_folders'].

    Returns:
        {class_label: [Path, ...]}
    """
    data_cfg = cfg.get("data", {})
    dataset_cfg = cfg.get("dataset", {})
    raw_dir = Path(data_cfg.get("raw_dir", dataset_cfg.get("path", "FruitGrade_Dataset")))
    skip = {s.lower() for s in data_cfg.get("skip_folders", [])}
    valid_exts = {e.lower() for e in data_cfg.get("image_extensions", [".jpg", ".jpeg", ".png"])}

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir.resolve()}")

    class_map: dict[str, list[Path]] = {}

    for fruit_dir in sorted(raw_dir.iterdir()):
        if not fruit_dir.is_dir():
            continue
        if fruit_dir.name.lower() in skip:
            logger.info(f"  Skipping folder: {fruit_dir.name}")
            continue

        for grade_dir in sorted(fruit_dir.iterdir()):
            if not grade_dir.is_dir():
                continue
            grade = grade_dir.name.strip()
            if grade.lower() not in VALID_GRADES:
                logger.info(f"  Skipping non-grade folder: {fruit_dir.name}/{grade_dir.name}")
                continue

            label = f"{fruit_dir.name}_{grade.upper()}"
            images: list[Path] = []

            for file in grade_dir.iterdir():
                if file.suffix.lower() in valid_exts:
                    images.append(file)

            if images:
                class_map.setdefault(label, []).extend(images)

    total = sum(len(v) for v in class_map.values())
    logger.info(f"Scan complete — {len(class_map)} classes, {total} total images")
    for label, paths in sorted(class_map.items()):
        logger.info(f"  {label}: {len(paths)} images")

    return class_map


# ─────────────────────────────────────────────
# 2. Clean corrupt / tiny images
# ─────────────────────────────────────────────

def clean_images(
    class_map: dict[str, list[Path]],
    min_size_bytes: int = 1024,
) -> dict[str, list[Path]]:
    """
    Filter out:
      - Files smaller than min_size_bytes (likely corrupt/empty)
      - Files that PIL cannot open/verify

    Returns:
        Cleaned class_map with bad files removed.
    """
    cleaned: dict[str, list[Path]] = {}
    removed_total = 0

    for label, paths in class_map.items():
        good: list[Path] = []
        for p in paths:
            # Size check
            if p.stat().st_size < min_size_bytes:
                logger.warning(f"  REMOVED (too small {p.stat().st_size}B): {p.name}")
                removed_total += 1
                continue
            # Integrity check
            try:
                with Image.open(p) as img:
                    img.verify()   # raises on corrupt header
                good.append(p)
            except Exception as exc:
                logger.warning(f"  REMOVED (corrupt): {p.name} — {exc}")
                removed_total += 1

        cleaned[label] = good

    total_after = sum(len(v) for v in cleaned.values())
    logger.info(f"Cleaning done — removed {removed_total} images, {total_after} remain")
    return cleaned


# ─────────────────────────────────────────────
# 3. Split and copy
# ─────────────────────────────────────────────

def split_and_copy(
    class_map: dict[str, list[Path]],
    cfg: dict,
    dry_run: bool = False,
) -> dict[str, dict[str, int]]:
    """
    Stratified per-class split → copy files into:
        ml/data/train/<label>/
        ml/data/val/<label>/
        ml/data/test/<label>/

    Originals in FruitGrade_Dataset/ are preserved (copy, not move).

    Args:
        class_map: {label: [image_paths]}
        cfg:       Loaded config dict.
        dry_run:   If True, skip actual file copying (preview only).

    Returns:
        Summary dict {label: {train: N, val: N, test: N}}
    """
    data_cfg = cfg.get("data", {})
    dataset_cfg = cfg.get("dataset", {})
    
    set_seed(data_cfg.get("split_seed", 42))

    train_root = Path(data_cfg.get("train_dir", dataset_cfg.get("path", "FruitGrade_Dataset")))
    val_root   = Path(data_cfg.get("val_dir", dataset_cfg.get("path", "FruitGrade_Dataset")))
    test_root  = Path(data_cfg.get("test_dir", dataset_cfg.get("path", "FruitGrade_Dataset")))

    train_r = data_cfg.get("train_ratio", 0.7)
    val_r   = data_cfg.get("val_ratio", 0.2)
    # test_r  = data_cfg.get("test_ratio", 0.1)  # implicit remainder

    summary: dict[str, dict[str, int]] = {}

    for label, paths in sorted(class_map.items()):
        if len(paths) < 3:
            logger.warning(f"  Skipping {label} — only {len(paths)} image(s), need ≥ 3")
            continue

        # First split: train vs (val + test)
        train_paths, valtest_paths = train_test_split(
            paths,
            train_size=train_r,
            random_state=data_cfg.get("split_seed", 42),
            shuffle=True,
        )

        # Second split: val vs test  (from the remaining portion)
        relative_val = val_r / (1.0 - train_r)
        val_paths, test_paths = train_test_split(
            valtest_paths,
            train_size=relative_val,
            random_state=data_cfg.get("split_seed", 42),
            shuffle=True,
        )

        summary[label] = {
            "train": len(train_paths),
            "val":   len(val_paths),
            "test":  len(test_paths),
        }

        logger.info(
            f"  {label}: train={len(train_paths)} | val={len(val_paths)} | test={len(test_paths)}"
        )

        if dry_run:
            continue

        splits = {
            "train": (train_root, train_paths),
            "val":   (val_root,   val_paths),
            "test":  (test_root,  test_paths),
        }

        for split_name, (root, split_paths) in splits.items():
            dest_dir = ensure_dir(root / label)
            for src in split_paths:
                dst = dest_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)

    if dry_run:
        logger.info("DRY RUN — no files were copied")
    else:
        logger.info("Split & copy complete ✓")

    return summary


# ─────────────────────────────────────────────
# 4. Full pipeline entry
# ─────────────────────────────────────────────

def prepare_dataset(cfg: dict, dry_run: bool = False) -> dict:
    """
    End-to-end dataset preparation:
      scan → clean → split & copy

    Args:
        cfg:     Config dict (from config.yaml).
        dry_run: Preview only, no file writes.

    Returns:
        Summary dict from split_and_copy().
    """
    logger.info("=" * 55)
    logger.info("STEP 1 — Scanning raw dataset")
    logger.info("=" * 55)
    class_map = scan_raw_dataset(cfg)

    logger.info("=" * 55)
    logger.info("STEP 2 — Cleaning corrupt images")
    logger.info("=" * 55)
    _data_cfg = cfg.get("data", {})
    class_map = clean_images(
        class_map,
        min_size_bytes=_data_cfg.get("min_file_size_bytes", 1024),
    )

    logger.info("=" * 55)
    logger.info("STEP 3 — Splitting and copying")
    logger.info("=" * 55)
    summary = split_and_copy(class_map, cfg, dry_run=dry_run)

    return summary


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FruitGrade dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview split without copying files",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    prepare_dataset(cfg, dry_run=args.dry_run)
