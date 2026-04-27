"""
scripts/train.py
-----------------
Single entry-point for the complete ML training pipeline.

Steps:
    1. Load config
    2. Prepare dataset (scan → clean → split into train/val/test)
    3. Build DataLoaders
    4. Build EfficientNet-B0 model
    5. Train with early stopping + checkpointing
    6. Evaluate on test set
    7. Print final metrics

Usage:
    # Full pipeline (dataset prep + train + evaluate)
    python scripts/train.py

    # Preview dataset split without copying files
    python scripts/train.py --dry-run

    # Skip dataset prep if already done
    python scripts/train.py --skip-prep

    # Custom config
    python scripts/train.py --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path

# ── Make sure project root is on sys.path ─────────────────────────────────
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.evaluation.evaluator import evaluate
from ml.models.model_builder import build_model
from ml.preprocessing.dataset_prep import prepare_dataset
from ml.training.data_loader import get_class_names, get_dataloaders
from ml.training.trainer import train
from ml.utils.helpers import (
    ensure_dir,
    get_device,
    load_yaml,
    set_seed,
    setup_logger,
)

logger = setup_logger("train")


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FruitGrade EfficientNet-B0 Training Pipeline"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config.yaml (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview dataset split without copying files or training",
    )
    parser.add_argument(
        "--skip-prep", action="store_true",
        help="Skip dataset preparation (use existing ml/data/train|val|test)",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training; run evaluation only (requires a saved checkpoint)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── 1. Load config ────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path.resolve()}")
        sys.exit(1)

    cfg = load_yaml(cfg_path)
    set_seed(cfg["data"]["split_seed"])
    device = get_device(cfg["inference"]["device"])

    logger.info("=" * 60)
    logger.info("FruitGrade ML Pipeline")
    logger.info(f"  Config : {cfg_path.resolve()}")
    logger.info(f"  Device : {device}")
    logger.info("=" * 60)

    # ── 2. Create output dirs ─────────────────────────────────────
    for d in [
        cfg["data"]["train_dir"],
        cfg["data"]["val_dir"],
        cfg["data"]["test_dir"],
        cfg["model"]["save_dir"],
    ]:
        ensure_dir(d)

    # ── 3. Dataset preparation ────────────────────────────────────
    if not args.skip_prep:
        logger.info("\n[PHASE 1] Dataset Preparation")
        summary = prepare_dataset(cfg, dry_run=args.dry_run)

        if args.dry_run:
            logger.info("Dry-run complete. No files copied. Exiting.")
            return

        logger.info(f"\nSplit summary:")
        total_t, total_v, total_te = 0, 0, 0
        for label, counts in sorted(summary.items()):
            logger.info(
                f"  {label:<20} train={counts['train']:>4} | "
                f"val={counts['val']:>3} | test={counts['test']:>3}"
            )
            total_t  += counts["train"]
            total_v  += counts["val"]
            total_te += counts["test"]
        logger.info(f"  {'TOTAL':<20} train={total_t:>4} | val={total_v:>3} | test={total_te:>3}")
    else:
        logger.info("[PHASE 1] Skipping dataset preparation (--skip-prep)")

    if args.skip_train:
        logger.info("[PHASE 2+3] Skipping training (--skip-train)")
    else:
        # ── 4. DataLoaders ────────────────────────────────────────
        logger.info("\n[PHASE 2] Building DataLoaders")
        dataloaders = get_dataloaders(cfg)

        if dataloaders["train"] is None:
            logger.error(
                "Train DataLoader is None. "
                "Run dataset preparation first (remove --skip-prep)."
            )
            sys.exit(1)

        # Sync num_classes with actual data
        n_classes = len(dataloaders["train"].dataset.classes)
        cfg["model"]["num_classes"] = n_classes
        logger.info(f"  num_classes set to {n_classes}")

        # ── 5. Model ──────────────────────────────────────────────
        logger.info("\n[PHASE 3] Building Model")
        model = build_model(cfg)

        # ── 6. Train ──────────────────────────────────────────────
        logger.info("\n[PHASE 4] Training")
        model = train(model, dataloaders, cfg, device)

    # ── 7. Evaluate ───────────────────────────────────────────────
    logger.info("\n[PHASE 5] Evaluation on Test Set")

    if args.skip_train:
        # Need to build model + load checkpoint for eval-only mode
        dataloaders = get_dataloaders(cfg)
        n_classes   = len(dataloaders["train"].dataset.classes)
        cfg["model"]["num_classes"] = n_classes
        model       = build_model(cfg)
        ckpt_path   = Path(cfg["model"]["save_dir"]) / cfg["model"]["checkpoint_name"]
        if not ckpt_path.exists():
            logger.error(f"No checkpoint found at {ckpt_path}. Cannot evaluate.")
            sys.exit(1)
        import torch
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

    class_names = get_class_names(cfg)
    eval_metrics = evaluate(
        model,
        dataloaders["test"],
        cfg,
        device,
        class_names=class_names,
    )

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Test Accuracy : {eval_metrics.get('accuracy', 'N/A')}%")
    logger.info(f"  Mean Conf     : {eval_metrics.get('mean_conf', 'N/A')}")
    logger.info(
        f"  Best checkpoint: "
        f"{Path(cfg['model']['save_dir']) / cfg['model']['checkpoint_name']}"
    )
    logger.info(
        f"  Confusion matrix: "
        f"{Path(cfg['model']['save_dir']) / 'confusion_matrix.png'}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
