"""
scripts/train_yolo.py
----------------------
Train YOLOv8 on the auto-labeled fruit detection dataset.

v3 -- CPU-SAFE edition:
  - Hardcoded CPU-safe defaults (batch=4, imgsz=416, workers=1, amp=False)
  - Dataset validation before training starts
  - Graceful crash recovery (saves even on KeyboardInterrupt)
  - Tamed augmentation (mosaic=0.7, mixup=0.05)

Prerequisites:
    1. Run scripts/prepare_yolo_dataset.py first to create yolo_dataset/
    2. Ensure ultralytics is installed: pip install ultralytics

Usage:
    python scripts/train_yolo.py
    python scripts/train_yolo.py --epochs 5 --batch 2
    python scripts/train_yolo.py --resume
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataset(data_dir: Path, max_check: int = 500) -> bool:
    """
    Quick-scan the YOLO dataset for common problems that cause training crashes.

    Checks:
        - Label files exist and are non-empty
        - Bbox values are in valid range (0-1)
        - No negative values
        - Corresponding image files exist

    Args:
        data_dir:  Root dataset directory (e.g. yolo_dataset/)
        max_check: Max number of label files to validate (for speed)

    Returns:
        True if dataset passes validation, False otherwise.
    """
    print("[VALIDATE] Checking dataset integrity...")

    errors = 0
    warnings = 0
    checked = 0

    for split in ["train", "val"]:
        lbl_dir = data_dir / split / "labels"
        img_dir = data_dir / split / "images"

        if not lbl_dir.exists():
            print(f"  [ERROR] Labels directory missing: {lbl_dir}")
            errors += 1
            continue

        if not img_dir.exists():
            print(f"  [ERROR] Images directory missing: {img_dir}")
            errors += 1
            continue

        label_files = sorted(lbl_dir.glob("*.txt"))
        if not label_files:
            print(f"  [ERROR] No label files in {lbl_dir}")
            errors += 1
            continue

        for lbl_path in label_files[:max_check]:
            checked += 1

            # Check label is non-empty
            content = lbl_path.read_text(encoding="utf-8").strip()
            if not content:
                print(f"  [ERROR] Empty label: {lbl_path.name}")
                errors += 1
                continue

            # Check corresponding image exists
            stem = lbl_path.stem
            img_found = False
            for ext in IMAGE_EXTS:
                if (img_dir / f"{stem}{ext}").exists():
                    img_found = True
                    break
            if not img_found:
                print(f"  [WARN] No image for label: {lbl_path.name}")
                warnings += 1

            # Validate bbox values
            for line_num, line in enumerate(content.split("\n"), 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"  [ERROR] Bad format in {lbl_path.name}:{line_num} "
                          f"(expected 5 values, got {len(parts)})")
                    errors += 1
                    continue

                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    print(f"  [ERROR] Non-numeric values in {lbl_path.name}:{line_num}")
                    errors += 1
                    continue

                # Check ranges
                if cls_id < 0 or cls_id > 20:
                    print(f"  [ERROR] Invalid class_id={cls_id} in {lbl_path.name}")
                    errors += 1
                if any(v < 0 or v > 1.0 for v in [xc, yc, w, h]):
                    print(f"  [ERROR] Bbox out of range in {lbl_path.name}: "
                          f"xc={xc} yc={yc} w={w} h={h}")
                    errors += 1

    print(f"  Checked : {checked} label files")
    print(f"  Errors  : {errors}")
    print(f"  Warnings: {warnings}")

    if errors > 0:
        print("[VALIDATE] FAILED -- fix errors before training")
        return False

    print("[VALIDATE] PASSED")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 fruit detector (v3 CPU-safe)")
    parser.add_argument(
        "--data",
        default=str(BASE_DIR / "yolo_dataset" / "data.yaml"),
        help="Path to data.yaml (default: yolo_dataset/data.yaml)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Pretrained YOLO weights (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs (default: 10 -- CPU optimized)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size (default: 4 -- CPU safe, prevents OOM)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
        help="Input image size (default: 416 -- faster on CPU than 640)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)",
    )
    parser.add_argument(
        "--name",
        default="fruit_detector_v1",
        help="Experiment name (default: fruit_detector_v1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip dataset validation (not recommended)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    data_yaml = Path(args.data)
    data_dir = data_yaml.parent  # yolo_dataset/

    if not data_yaml.exists():
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        print("        Run scripts/prepare_yolo_dataset.py first.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.")
        print("        Run: pip install ultralytics")
        sys.exit(1)

    # ── Header ────────────────────────────────────────────────────
    print("=" * 60)
    print("YOLOv8 Fruit Detector Training (v3 CPU-SAFE)")
    print("=" * 60)
    print(f"  Data     : {data_yaml}")
    print(f"  Model    : {args.model}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"  ImgSize  : {args.imgsz}")
    print(f"  Patience : {args.patience}")
    print(f"  Name     : {args.name}")
    print(f"  Device   : cpu (forced)")
    print(f"  AMP      : disabled (CPU safe)")
    print(f"  Workers  : 1 (prevents multiprocessing crash)")
    print()
    print("  Augmentation (balanced for CPU):")
    print("    HSV hue       : 0.015")
    print("    HSV saturation: 0.70")
    print("    HSV value     : 0.40")
    print("    Flip LR       : 0.50")
    print("    Translate     : 0.10")
    print("    Scale         : 0.50")
    print("    Mosaic        : 0.70 (reduced)")
    print("    Mixup         : 0.05 (minimal)")
    print("=" * 60)
    print()

    # ── Step 1: Validate dataset ──────────────────────────────────
    if not args.skip_validation:
        ok = validate_dataset(data_dir)
        if not ok:
            print("\n[ABORT] Dataset validation failed. Fix errors above.")
            print("        Or use --skip-validation to bypass (not recommended).")
            sys.exit(1)
        print()

    # ── Step 2: Load model ────────────────────────────────────────
    print("[LOADING] YOLOv8 nano model...")
    model = YOLO(args.model)
    print("[OK] Model loaded.\n")

    # ── Step 3: Train ─────────────────────────────────────────────
    # All settings hardcoded for CPU stability.
    # DO NOT increase batch or imgsz without GPU.
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        name=args.name,
        verbose=True,
        save=True,
        plots=True,
        exist_ok=True,           # overwrite previous run with same name

        # -- CPU-CRITICAL settings ---------------------------------
        device="cpu",            # FORCE CPU (no GPU probing)
        amp=False,               # disable mixed precision (CPU doesn't support it well)
        workers=1,               # minimal data loading workers (prevents fork crash on Windows)

        # -- Balanced augmentation (not too aggressive) ------------
        hsv_h=0.015,             # hue shift
        hsv_s=0.70,              # saturation shift
        hsv_v=0.40,              # brightness shift

        fliplr=0.50,             # horizontal flip
        flipud=0.0,              # no vertical flip
        translate=0.10,          # position shift
        scale=0.50,              # scale variation

        mosaic=0.70,             # mosaic (reduced from 1.0 to save memory)
        mixup=0.05,              # mixup (minimal to avoid memory spikes)

        # -- Regularization ----------------------------------------
        weight_decay=0.0005,
        warmup_epochs=2.0,       # short warmup (training is only 10 epochs)
    )

    if args.resume:
        train_kwargs["resume"] = True

    print("[TRAINING] Starting... (this will take a while on CPU)")
    print(f"           Estimated: ~{args.epochs * 10}-{args.epochs * 20} minutes")
    print()

    t_start = time.time()

    try:
        results = model.train(**train_kwargs)
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("[INTERRUPTED] Training stopped by user.")
        print("              Partial weights may have been saved.")
        print("              Check: runs/detect/{}/weights/".format(args.name))
        print("=" * 60)
        # Still try to copy whatever was saved
        _copy_best_weights(args.name)
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] Training failed: {exc}")
        print("        Check error above for details.")
        _copy_best_weights(args.name)
        sys.exit(1)

    elapsed = time.time() - t_start
    elapsed_min = elapsed / 60

    # ── Step 4: Post-training ─────────────────────────────────────
    print()
    print("=" * 60)
    print(f"[DONE] Training complete! ({elapsed_min:.1f} minutes)")
    print()
    _copy_best_weights(args.name)
    print("=" * 60)


def _copy_best_weights(run_name: str) -> None:
    """Copy best.pt to the project models directory if it exists."""
    # Try multiple possible locations
    candidates = [
        Path(f"runs/detect/{run_name}/weights/best.pt"),
        Path(f"runs/detect/{run_name}2/weights/best.pt"),
        Path(f"runs/detect/{run_name}3/weights/best.pt"),
    ]

    for best_pt in candidates:
        if best_pt.exists():
            dest = BASE_DIR / "ml" / "models" / "saved" / "yolo_fruit_best.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_pt, dest)
            print(f"  Best weights : {best_pt}")
            print(f"  Copied to    : {dest}")
            print()
            print("  System will auto-detect the new model on restart.")
            print("  Run: python router_main.py")
            return

    # Also check for last.pt as fallback
    last_pt = Path(f"runs/detect/{run_name}/weights/last.pt")
    if last_pt.exists():
        dest = BASE_DIR / "ml" / "models" / "saved" / "yolo_fruit_best.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(last_pt, dest)
        print(f"  [WARN] best.pt not found, using last.pt instead")
        print(f"  Last weights : {last_pt}")
        print(f"  Copied to    : {dest}")
        return

    print("  [WARN] No trained weights found to copy.")
    print("         Training may not have completed any full epoch.")


if __name__ == "__main__":
    main()
