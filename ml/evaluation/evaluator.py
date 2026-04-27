"""
ml/evaluation/evaluator.py
---------------------------
Evaluates a trained model on the test split.

Outputs:
  - Overall accuracy
  - Per-class precision / recall / F1 (classification_report)
  - Confusion matrix saved as PNG
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.utils.helpers import ensure_dir, setup_logger

logger = setup_logger(__name__)


# ─────────────────────────────────────────────
# Inference pass
# ─────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[float]]:
    """
    Run inference over all batches in loader.

    Returns:
        all_labels:    Ground-truth class indices
        all_preds:     Predicted class indices
        all_confs:     Max softmax probability for each prediction
    """
    model.eval()
    all_labels: list[int] = []
    all_preds:  list[int] = []
    all_confs:  list[float] = []

    for images, labels in tqdm(loader, desc="  [test] ", unit="batch", leave=False):
        images = images.to(device)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)
        confs, preds = probs.max(dim=1)

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_confs.extend(confs.cpu().tolist())

    return all_labels, all_preds, all_confs


# ─────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    labels: list[int],
    preds: list[int],
    class_names: list[str],
    save_path: Path,
) -> None:
    """
    Save a seaborn confusion-matrix heatmap.

    Args:
        labels:       Ground-truth class indices.
        preds:        Predicted class indices.
        class_names:  Ordered list of class names.
        save_path:    Output PNG path.
    """
    cm = confusion_matrix(labels, preds)
    n  = len(class_names)

    fig_size = max(10, n // 2)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title("Confusion Matrix — FruitGrade Classifier", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()

    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────
# Full evaluation entry point
# ─────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict:
    """
    Run full test-set evaluation and report metrics.

    Args:
        model:        Trained model (already on device).
        test_loader:  DataLoader for the test split.
        cfg:          Config dict.
        device:       Torch device.
        class_names:  If None, attempts to read from test_loader.dataset.

    Returns:
        Dict with keys: accuracy, report, per_class_f1
    """
    if test_loader is None:
        logger.warning("Test loader is None — skipping evaluation.")
        return {}

    # Resolve class names
    if class_names is None:
        try:
            class_names = test_loader.dataset.classes  # type: ignore[union-attr]
        except AttributeError:
            class_names = [str(i) for i in range(cfg["model"]["num_classes"])]

    logger.info("=" * 55)
    logger.info("EVALUATION — Test Set")
    logger.info("=" * 55)

    labels, preds, confs = run_inference(model, test_loader, device)

    # ── Overall accuracy ──────────────────────────────────────────
    accuracy = 100.0 * sum(p == l for p, l in zip(preds, labels)) / len(labels)
    logger.info(f"  Test Accuracy: {accuracy:.2f}%")
    logger.info(f"  Mean Confidence: {np.mean(confs):.4f}")

    # ── Classification report ─────────────────────────────────────
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    logger.info(f"\n{report}")

    # ── Per-class F1 dict ─────────────────────────────────────────
    report_dict = classification_report(
        labels, preds, target_names=class_names, output_dict=True
    )
    per_class_f1 = {
        cls: round(report_dict[cls]["f1-score"], 4)
        for cls in class_names
        if cls in report_dict
    }

    # ── Confusion matrix ──────────────────────────────────────────
    cm_path = Path(cfg["model"]["save_dir"]) / "confusion_matrix.png"
    plot_confusion_matrix(labels, preds, class_names, cm_path)

    return {
        "accuracy":      round(accuracy, 4),
        "report":        report,
        "per_class_f1":  per_class_f1,
        "mean_conf":     round(float(np.mean(confs)), 4),
    }
