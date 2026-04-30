"""
scripts/run_inference.py
------------------------
End-to-end inference test script.

Loads the trained EfficientNet-B0 + YOLOv8 pipeline and runs predictions on
every image inside test_images/, printing results and saving annotated images
to test_outputs/.

Usage:
    python scripts/run_inference.py
    python scripts/run_inference.py --config configs/config.yaml
    python scripts/run_inference.py --images path/to/other/folder
"""

import argparse
import io
import json
import sys
from pathlib import Path

# ── Force UTF-8 stdout on Windows so emoji / box chars don't crash ───────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Fix sys.path so all ml.* imports work regardless of cwd ──────────────────
BASE_DIR = Path(__file__).resolve().parents[1]   # project root
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ── Project imports ───────────────────────────────────────────────────────────
from ml.pipeline.inference_pipeline import InferencePipeline
from ml.preprocessing.dataset_prep import prepare_dataset
from ml.utils.helpers import ensure_dir, load_yaml, resolve_cfg_paths, setup_logger

logger = setup_logger("run_inference")

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FruitGrade inference on a folder of images"
    )
    parser.add_argument(
        "--config",
        default=str(BASE_DIR / "configs" / "config.yaml"),
        help="Path to config.yaml (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--images",
        default=str(BASE_DIR / "test_images"),
        help="Folder with input images (default: test_images/)",
    )
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "test_outputs"),
        help="Folder for annotated output images (default: test_outputs/)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Dataset guard — auto-prep if splits missing
# ─────────────────────────────────────────────

def ensure_splits_exist(cfg: dict) -> None:
    """
    Check that ml/data/train, val, test all exist and are non-empty.
    If any split is missing, run dataset preparation automatically
    (copies images only, does NOT retrain the model).
    """
    train_dir = Path(cfg["data"]["train_dir"])
    val_dir   = Path(cfg["data"]["val_dir"])
    test_dir  = Path(cfg["data"]["test_dir"])

    splits_ok = (
        train_dir.exists() and any(train_dir.iterdir()) and
        val_dir.exists()   and any(val_dir.iterdir())   and
        test_dir.exists()  and any(test_dir.iterdir())
    )

    if splits_ok:
        logger.info("[OK] Dataset splits found -- skipping preparation")
        return

    logger.warning(
        "[WARN] One or more dataset splits are missing. "
        "Running dataset preparation (no retraining)..."
    )
    raw_dir = Path(cfg["data"]["raw_dir"])
    if not raw_dir.exists():
        logger.error(
            f"[ERROR] Raw dataset not found at: {raw_dir}\n"
            "   Please ensure FruitGrade_Dataset/ is in the project root."
        )
        sys.exit(1)

    prepare_dataset(cfg, dry_run=False)
    logger.info("[OK] Dataset splits created successfully")


# ─────────────────────────────────────────────
# Model checkpoint guard
# ─────────────────────────────────────────────

def check_model_exists(cfg: dict) -> Path:
    """
    Verify that the trained checkpoint exists.
    Returns the absolute Path to the checkpoint.
    """
    ckpt_path = Path(cfg["model"]["save_dir"]) / cfg["model"]["checkpoint_name"]
    if not ckpt_path.exists():
        logger.error(
            f"[ERROR] Model checkpoint not found: {ckpt_path}\n"
            "   Train the model first:\n"
            "       python scripts/train.py"
        )
        sys.exit(1)
    logger.info(f"[OK] Checkpoint found: {ckpt_path}")
    return ckpt_path


# ─────────────────────────────────────────────
# Pretty result printer
# ─────────────────────────────────────────────

def _print_result(img_name: str, results: list[dict]) -> None:
    """Print a compact, structured result for each detection."""
    if not results:
        print("  [WARN] No results returned (YOLO fallback disabled and no detections)")
        return

    for i, r in enumerate(results, start=1):
        # Core fields requested in the task spec
        summary = {
            "fruit":      r.get("fruit", "unknown"),
            "grade":      r.get("grade"),
            "confidence": r.get("confidence", 0.0),
            "detection":  r.get("detection", "unknown"),
        }
        # Extra useful fields
        extra = {
            "confidence_level": r.get("confidence_level"),
            "unknown":          r.get("unknown"),
            "explanation":      r.get("explanation"),
            "bbox":             r.get("bbox"),
            "yolo_conf":        r.get("yolo_conf"),
            "processing_ms":    r.get("processing_time_ms"),
        }
        tag = f"[Detection {i}]" if len(results) > 1 else ""
        if tag:
            print(f"  {tag}")
        print(f"  {json.dumps(summary, indent=4)}")
        print(f"  Top-3 classes: {r.get('top_k', [])}")
        print(f"  Extra: {json.dumps(extra, indent=4)}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # -- 1. Load + resolve config -------------------------------------------
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"[ERROR] Config not found: {cfg_path}")
        sys.exit(1)

    cfg = load_yaml(cfg_path)
    cfg = resolve_cfg_paths(cfg, BASE_DIR)   # make ALL paths absolute

    logger.info("=" * 60)
    logger.info("FruitGrade Inference Pipeline")
    logger.info(f"  Config  : {cfg_path}")
    logger.info(f"  Images  : {args.images}")
    logger.info(f"  Outputs : {args.output}")
    logger.info("=" * 60)

    # -- 2. Ensure dataset splits exist (needed for class_names) ------------
    ensure_splits_exist(cfg)

    # -- 3. Ensure model checkpoint exists ----------------------------------
    check_model_exists(cfg)

    # -- 4. Build inference pipeline ----------------------------------------
    logger.info("\nInitialising InferencePipeline...")
    pipeline = InferencePipeline(cfg)
    logger.info("[OK] Pipeline ready\n")

    # -- 5. Gather input images ---------------------------------------------
    img_folder  = Path(args.images)
    out_folder  = ensure_dir(Path(args.output))

    if not img_folder.exists():
        logger.error(
            f"[ERROR] Image folder not found: {img_folder}\n"
            "   Create test_images/ and add some fruit images."
        )
        sys.exit(1)

    images = [p for p in sorted(img_folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS]

    if not images:
        logger.error(
            f"[ERROR] No images found in: {img_folder}\n"
            f"   Supported formats: {', '.join(IMAGE_EXTS)}"
        )
        sys.exit(1)

    logger.info(f"Found {len(images)} image(s) in {img_folder}\n")

    # -- 6. Run inference on each image ------------------------------------
    success_count = 0
    error_count   = 0

    for img_path in images:
        vis_path = out_folder / f"out_{img_path.name}"
        print("-" * 60)
        print(f"[IMAGE] {img_path.name}")

        try:
            results = pipeline.run(
                source=img_path,
                visualize=True,
                vis_save_path=vis_path,
            )
            _print_result(img_path.name, results)
            print(f"  [SAVED] {vis_path}")
            success_count += 1

        except Exception as exc:
            logger.exception(f"[ERROR] processing {img_path.name}: {exc}")
            error_count += 1

    # -- 7. Summary --------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"[DONE] {success_count} succeeded, {error_count} failed")
    print(f"   Annotated images saved to: {out_folder}")
    print("=" * 60)


if __name__ == "__main__":
    main()
