"""
scripts/prepare_yolo_dataset.py
--------------------------------
Converts the classification-only FruitGrade_Dataset into a YOLO-format
object detection dataset with auto-generated bounding box labels.

v2 improvements:
  - RANDOMIZED bounding boxes (position jitter + scale variation)
  - Debug visualization (saves 20 sample images with drawn bboxes)
  - Better simulation of real-world object positions

Since every image contains ONE fruit that is centered and fills ~80-90%
of the frame, we generate one label per image with slight randomization
to improve YOLO generalization:

    class_id  (0.5 +/- 0.1)  (0.5 +/- 0.1)  (0.6-0.9)  (0.6-0.9)

Usage:
    python scripts/prepare_yolo_dataset.py
    python scripts/prepare_yolo_dataset.py --source FruitGrade_Dataset --output yolo_dataset
    python scripts/prepare_yolo_dataset.py --no-jitter     # disable randomization
    python scripts/prepare_yolo_dataset.py --dry-run       # preview without copying
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]

# ── Class mapping (fruit name -> YOLO class id) ──────────────────────────────
# Grade (A/B/C) is IGNORED for detection -- YOLO only needs to know "this is an apple".
# The classifier downstream handles grade prediction.
CLASS_MAP: dict[str, int] = {
    "apple":      0,
    "apricot":    1,
    "banana":     2,
    "grapes":     3,
    "guava":      4,
    "honeydew":   5,
    "orange":     6,
    "tangerine":  7,
    "watermelon": 8,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Bbox generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_bbox(
    rng: random.Random,
    bbox_base: float = 0.85,
    jitter: bool = True,
) -> tuple[float, float, float, float]:
    """
    Generate a single YOLO bounding box (normalized 0-1).

    Args:
        rng:       Random instance for reproducibility.
        bbox_base: Base bounding box size (used when jitter=False).
        jitter:    If True, add random position shift + scale variation.

    Returns:
        (x_center, y_center, width, height) -- all clamped to valid range.
    """
    if not jitter:
        return (0.5, 0.5, bbox_base, bbox_base)

    # Random position shift: center +/- 0.10
    x_center = 0.5 + rng.uniform(-0.10, 0.10)
    y_center = 0.5 + rng.uniform(-0.10, 0.10)

    # Random scale: width/height between 0.60 and 0.90
    w = rng.uniform(0.60, 0.90)
    h = rng.uniform(0.60, 0.90)

    # Ensure bbox stays inside image (0-1 range)
    # Left edge = x_center - w/2 >= 0  =>  x_center >= w/2
    # Right edge = x_center + w/2 <= 1  =>  x_center <= 1 - w/2
    x_center = max(w / 2, min(1.0 - w / 2, x_center))
    y_center = max(h / 2, min(1.0 - h / 2, y_center))

    return (
        round(x_center, 4),
        round(y_center, 4),
        round(w, 4),
        round(h, 4),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Debug visualization
# ─────────────────────────────────────────────────────────────────────────────

def save_debug_samples(
    output_dir: Path,
    sample_pairs: list[tuple[Path, str, str]],
    max_samples: int = 20,
) -> None:
    """
    Save sample images with bounding boxes drawn on them for visual verification.

    Args:
        output_dir:    Root output directory (debug_samples/ will be created inside).
        sample_pairs:  List of (image_path, label_line, dst_filename) tuples.
        max_samples:   Max number of debug images to save.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[WARN] Pillow not installed, skipping debug visualization.")
        return

    debug_dir = output_dir / "debug_samples"
    debug_dir.mkdir(parents=True, exist_ok=True)

    reverse_class_map = {v: k for k, v in CLASS_MAP.items()}
    saved = 0

    for img_path, label_line, dst_name in sample_pairs:
        if saved >= max_samples:
            break

        try:
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            w_img, h_img = img.size

            # Parse YOLO label: class_id x_center y_center width height
            parts = label_line.strip().split()
            cls_id = int(parts[0])
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Convert normalized coords to pixel coords
            x1 = int((xc - bw / 2) * w_img)
            y1 = int((yc - bh / 2) * h_img)
            x2 = int((xc + bw / 2) * w_img)
            y2 = int((yc + bh / 2) * h_img)

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)

            # Draw label text
            fruit_name = reverse_class_map.get(cls_id, str(cls_id))
            label_text = f"{fruit_name} ({cls_id})"
            draw.text((x1 + 4, y1 + 4), label_text, fill="lime")

            # Save
            out_path = debug_dir / f"debug_{saved:02d}_{dst_name}.jpg"
            img.save(out_path, quality=90)
            saved += 1

        except Exception as e:
            print(f"[WARN] Debug save failed for {img_path.name}: {e}")
            continue

    print(f"  Saved {saved} debug samples to: {debug_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def discover_images(source_dir: Path) -> list[tuple[Path, int]]:
    """
    Walk FruitGrade_Dataset/<fruit>/<grade>/*.jpg and return a list of
    (image_path, class_id) tuples.

    Folder structure expected:
        source_dir/
            apple/A/*.jpg
            apple/B/*.jpg
            banana/A/*.jpg
            ...
    """
    pairs: list[tuple[Path, int]] = []
    skipped_fruits: set[str] = set()

    for fruit_dir in sorted(source_dir.iterdir()):
        if not fruit_dir.is_dir():
            continue

        fruit_name = fruit_dir.name.lower()
        class_id = CLASS_MAP.get(fruit_name)

        if class_id is None:
            skipped_fruits.add(fruit_dir.name)
            continue

        # Iterate over grade sub-folders (A, B, C)
        for grade_dir in sorted(fruit_dir.iterdir()):
            if not grade_dir.is_dir():
                continue

            for img_path in sorted(grade_dir.iterdir()):
                if img_path.suffix.lower() in IMAGE_EXTS:
                    pairs.append((img_path, class_id))

    if skipped_fruits:
        print(f"[WARN] Skipped unknown fruit folders: {skipped_fruits}")

    return pairs


def split_dataset(
    pairs: list[tuple[Path, int]],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Shuffle and split into train / val / test sets.
    Remaining after train + val goes to test.
    """
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    return train, val, test


def write_split(
    split_pairs: list[tuple[Path, int]],
    output_dir: Path,
    split_name: str,
    bbox_base: float,
    jitter: bool,
    seed: int,
    dry_run: bool = False,
    collect_debug: list | None = None,
) -> int:
    """
    Copy images and create YOLO .txt label files for one split.

    Args:
        collect_debug: If provided, appends (img_path, label_line, stem) tuples
                       for debug visualization.

    Returns the number of images processed.
    """
    img_dir = output_dir / split_name / "images"
    lbl_dir = output_dir / split_name / "labels"

    if not dry_run:
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    # Separate RNG per split so results are reproducible
    rng = random.Random(seed + hash(split_name))

    count = 0
    seen_names: dict[str, int] = {}

    for img_path, class_id in split_pairs:
        # ── Generate unique filename ──────────────────────────────
        # Some images across different grades may have the same filename.
        # Prefix with fruit name and grade to avoid collisions.
        fruit_name = img_path.parent.parent.name.lower()
        grade = img_path.parent.name.upper()
        stem = f"{fruit_name}_{grade}_{img_path.stem}"

        # Handle remaining collisions (e.g. augmented copies)
        if stem in seen_names:
            seen_names[stem] += 1
            stem = f"{stem}_{seen_names[stem]}"
        else:
            seen_names[stem] = 0

        img_ext = img_path.suffix
        dst_img = img_dir / f"{stem}{img_ext}"
        dst_lbl = lbl_dir / f"{stem}.txt"

        # ── Generate YOLO bbox (with optional jitter) ─────────────
        x_center, y_center, w, h = generate_bbox(rng, bbox_base, jitter)
        label_line = f"{class_id} {x_center} {y_center} {w} {h}\n"

        if dry_run:
            if count < 5:  # Show first 5 as preview
                print(f"  [DRY] {img_path.name} -> {dst_img.name}")
                print(f"         label: {label_line.strip()}")
        else:
            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text(label_line, encoding="utf-8")

        # Collect for debug visualization
        if collect_debug is not None and count < 30:
            collect_debug.append((img_path, label_line, stem))

        count += 1

    return count


def write_data_yaml(output_dir: Path, dry_run: bool = False) -> None:
    """Generate YOLO data.yaml configuration file."""
    yaml_content = f"""# --------------------------------------------------
#  YOLO Fruit Detection -- Auto-generated data.yaml
# --------------------------------------------------
# Generated by: scripts/prepare_yolo_dataset.py (v2)
# Dataset: FruitGrade_Dataset (classification -> detection)
#
# Each image contains ONE centered fruit.
# Bounding boxes auto-generated with random jitter for generalization.
# Grade (A/B/C) is IGNORED -- YOLO only detects fruit type.
# The downstream EfficientNet classifier handles grade prediction.

path: {output_dir.resolve().as_posix()}
train: train/images
val: val/images
test: test/images

nc: {len(CLASS_MAP)}
names:
{chr(10).join(f'  {v}: {k}' for k, v in sorted(CLASS_MAP.items(), key=lambda x: x[1]))}
"""

    if dry_run:
        print("\n[DRY] data.yaml would contain:")
        print(yaml_content)
    else:
        yaml_path = output_dir / "data.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")
        print(f"[OK] Created: {yaml_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert classification dataset to YOLO detection format (v2)"
    )
    parser.add_argument(
        "--source",
        default=str(BASE_DIR / "FruitGrade_Dataset"),
        help="Path to source classification dataset (default: FruitGrade_Dataset/)",
    )
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "yolo_dataset"),
        help="Path for output YOLO dataset (default: yolo_dataset/)",
    )
    parser.add_argument(
        "--bbox-size",
        type=float,
        default=0.85,
        help="Base bounding box size (default: 0.85, used when --no-jitter)",
    )
    parser.add_argument(
        "--no-jitter",
        action="store_true",
        help="Disable bbox randomization (use fixed center 0.85 like v1)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Fraction of data for training (default: 0.80)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Fraction of data for validation (default: 0.10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip-debug",
        action="store_true",
        help="Skip saving debug visualization samples",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview output without creating files",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    use_jitter = not args.no_jitter

    if not source_dir.exists():
        print(f"[ERROR] Source dataset not found: {source_dir}")
        sys.exit(1)

    print("=" * 60)
    print("YOLO Dataset Preparation (v2 -- with jitter)")
    print("=" * 60)
    print(f"  Source       : {source_dir}")
    print(f"  Output       : {output_dir}")
    print(f"  BBox base    : {args.bbox_size}")
    print(f"  BBox jitter  : {use_jitter}")
    if use_jitter:
        print(f"    Position   : center +/- 0.10")
        print(f"    Scale      : 0.60 - 0.90")
    print(f"  Train/Val    : {args.train_ratio}/{args.val_ratio}/{round(1 - args.train_ratio - args.val_ratio, 2)}")
    print(f"  Seed         : {args.seed}")
    print(f"  Debug samples: {not args.skip_debug}")
    print(f"  Dry run      : {args.dry_run}")
    print()

    # ── Step 1: Discover all images ───────────────────────────────
    print("[1/5] Scanning dataset...")
    pairs = discover_images(source_dir)

    if not pairs:
        print("[ERROR] No images found in source dataset.")
        sys.exit(1)

    # Count per class
    class_counts: dict[int, int] = {}
    for _, cid in pairs:
        class_counts[cid] = class_counts.get(cid, 0) + 1

    print(f"  Found {len(pairs)} images across {len(class_counts)} classes:")
    for fruit, cid in sorted(CLASS_MAP.items(), key=lambda x: x[1]):
        count = class_counts.get(cid, 0)
        print(f"    [{cid}] {fruit:12s} -> {count} images")
    print()

    # ── Step 2: Split dataset ─────────────────────────────────────
    print("[2/5] Splitting dataset...")
    train, val, test = split_dataset(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"  Train : {len(train)} images")
    print(f"  Val   : {len(val)} images")
    print(f"  Test  : {len(test)} images")
    print()

    # ── Step 3: Write splits ──────────────────────────────────────
    print("[3/5] Writing dataset splits (with jitter)..." if use_jitter
          else "[3/5] Writing dataset splits (fixed center)...")

    if not args.dry_run and output_dir.exists():
        print(f"  [WARN] Output directory exists: {output_dir}")
        print(f"         Removing and recreating...")
        shutil.rmtree(output_dir)

    debug_samples: list[tuple[Path, str, str]] = []

    n_train = write_split(
        train, output_dir, "train", args.bbox_size,
        jitter=use_jitter, seed=args.seed,
        dry_run=args.dry_run, collect_debug=debug_samples,
    )
    n_val = write_split(
        val, output_dir, "val", args.bbox_size,
        jitter=use_jitter, seed=args.seed,
        dry_run=args.dry_run,
    )
    n_test = write_split(
        test, output_dir, "test", args.bbox_size,
        jitter=use_jitter, seed=args.seed,
        dry_run=args.dry_run,
    )

    print(f"  Train : {n_train} images + labels")
    print(f"  Val   : {n_val} images + labels")
    print(f"  Test  : {n_test} images + labels")
    print()

    # ── Step 4: Debug visualization ───────────────────────────────
    if not args.dry_run and not args.skip_debug:
        print("[4/5] Saving debug visualization samples...")
        save_debug_samples(output_dir, debug_samples, max_samples=20)
    else:
        print("[4/5] Skipping debug visualization.")
    print()

    # ── Step 5: Generate data.yaml ────────────────────────────────
    print("[5/5] Generating data.yaml...")
    write_data_yaml(output_dir, args.dry_run)

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    if args.dry_run:
        print("[DRY RUN] No files were created.")
        print("          Remove --dry-run to create the dataset.")
    else:
        print("[DONE] YOLO dataset ready!")
        print()
        print("Next steps:")
        print()
        print("  1. Check debug images:")
        print(f"     {output_dir / 'debug_samples'}")
        print()
        print("  2. Train YOLO:")
        print("     python scripts/train_yolo.py")
        print()
        print("  3. After training, restart Flask:")
        print("     python router_main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
