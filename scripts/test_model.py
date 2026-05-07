"""
scripts/test_model.py
---------------------
SAFE TESTING MODE for comparing model_v2 vs model_v3.

Features:
  - Test any model version without modifying config.yaml
  - Side-by-side comparison on the same images
  - Confidence distribution analysis
  - Per-class accuracy breakdown
  - UNKNOWN rate comparison
  - No permanent changes to production files

Usage:
    # Test v2 (production model):
    .venv\\Scripts\\python.exe scripts/test_model.py --model v2

    # Test v3 (fine-tuned model):
    .venv\\Scripts\\python.exe scripts/test_model.py --model v3

    # Compare both side-by-side:
    .venv\\Scripts\\python.exe scripts/test_model.py --compare

    # Test with specific images:
    .venv\\Scripts\\python.exe scripts/test_model.py --model v2 --images path/to/img1.jpg path/to/img2.jpg

This script does NOT modify:
  - config.yaml
  - model_v2.pth or model_v3.pth
  - Any backend, API, or controller files
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Fix imports
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ml.models.model_builder import build_model
from ml.utils.helpers import load_yaml, setup_logger, load_class_mapping

logger = setup_logger("test_model")


def load_model(model_path, class_names, device="cpu"):
    """Load a model from a .pth file."""
    cfg = {"model": {"num_classes": len(class_names)}, "inference": {"device": device}}
    model = build_model(cfg, num_classes=len(class_names))

    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_transform():
    """Standard inference transform matching training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_image(model, image_path, class_names, transform, threshold=0.40, device="cpu"):
    """Predict a single image and return structured result."""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)

    confidence = float(probs.max().item())
    pred_idx = int(probs.argmax().item())
    label = class_names[pred_idx]

    top_k_probs, top_k_idxs = probs.topk(3)
    top_k = [{"label": class_names[int(i)], "prob": round(float(p), 4)}
             for p, i in zip(top_k_probs.tolist(), top_k_idxs.tolist())]

    if confidence < threshold:
        return {"label": "unknown", "confidence": confidence, "unknown": True, "top_k": top_k}
    
    parts = label.rsplit("_", 1)
    fruit = parts[0] if len(parts) == 2 else label
    grade = parts[1] if len(parts) == 2 else "?"

    return {
        "label": label, "fruit": fruit, "grade": grade,
        "confidence": round(confidence, 4), "unknown": False, "top_k": top_k
    }


def get_sample_images(dataset_dir, max_per_class=3):
    """Get a few sample images from each class for testing."""
    samples = []
    for fruit_dir in sorted(Path(dataset_dir).iterdir()):
        if not fruit_dir.is_dir() or fruit_dir.name.lower() == "real":
            continue
        for grade_dir in sorted(fruit_dir.iterdir()):
            if not grade_dir.is_dir():
                continue
            imgs = [f for f in grade_dir.iterdir()
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
            for img in imgs[:max_per_class]:
                expected = f"{fruit_dir.name}_{grade_dir.name}"
                samples.append((img, expected))
    return samples


def test_model(model_path, model_name, class_names, threshold=0.40, samples=None, dataset_dir=None):
    """Test a single model and print results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING MODEL: {model_name}")
    logger.info(f"  Path: {model_path}")
    logger.info(f"  Threshold: {threshold}")
    logger.info(f"{'='*60}")

    model = load_model(model_path, class_names)
    transform = get_transform()

    if samples is None:
        if dataset_dir is None:
            dataset_dir = "FruitGrade_Dataset"
        samples = get_sample_images(dataset_dir, max_per_class=2)

    total = 0
    correct = 0
    unknown_count = 0
    fruit_correct = 0  # correct fruit type even if grade wrong
    per_class_correct = {}
    per_class_total = {}
    conf_sum = 0.0

    t_start = time.time()

    for img_path, expected_label in samples:
        result = predict_image(model, img_path, class_names, transform, threshold)
        total += 1
        conf_sum += result["confidence"]

        expected_fruit = expected_label.rsplit("_", 1)[0]

        if result["unknown"]:
            unknown_count += 1
            status = "UNKNOWN"
        elif result["label"] == expected_label:
            correct += 1
            fruit_correct += 1
            status = "CORRECT"
        elif result.get("fruit", "") == expected_fruit:
            fruit_correct += 1
            status = f"GRADE-WRONG ({result['label']})"
        else:
            status = f"WRONG ({result['label']})"

        per_class_total[expected_label] = per_class_total.get(expected_label, 0) + 1
        if result["label"] == expected_label:
            per_class_correct[expected_label] = per_class_correct.get(expected_label, 0) + 1

    elapsed = time.time() - t_start

    logger.info(f"\nRESULTS ({model_name}):")
    logger.info(f"  Total images: {total}")
    logger.info(f"  Exact match:  {correct}/{total} ({100*correct/total:.1f}%)" if total > 0 else "  No images")
    logger.info(f"  Fruit match:  {fruit_correct}/{total} ({100*fruit_correct/total:.1f}%)" if total > 0 else "")
    logger.info(f"  Unknown:      {unknown_count}/{total} ({100*unknown_count/total:.1f}%)" if total > 0 else "")
    logger.info(f"  Avg confidence: {conf_sum/total:.4f}" if total > 0 else "")
    logger.info(f"  Time: {elapsed:.1f}s ({1000*elapsed/total:.0f}ms/image)" if total > 0 else "")

    # Per-class breakdown
    logger.info(f"\nPer-class accuracy:")
    for cls in sorted(per_class_total.keys()):
        ct = per_class_total[cls]
        cc = per_class_correct.get(cls, 0)
        logger.info(f"  {cls:25s}  {cc}/{ct}  ({100*cc/ct:.0f}%)")

    return {
        "model": model_name,
        "total": total,
        "correct": correct,
        "fruit_correct": fruit_correct,
        "unknown": unknown_count,
        "avg_conf": conf_sum / total if total > 0 else 0,
        "accuracy": correct / total if total > 0 else 0,
        "fruit_accuracy": fruit_correct / total if total > 0 else 0,
        "unknown_rate": unknown_count / total if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Safe model testing")
    parser.add_argument("--model", choices=["v2", "v3"], default="v2",
                        help="Which model to test (default: v2)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare v2 and v3 side-by-side")
    parser.add_argument("--threshold", type=float, default=0.40,
                        help="Confidence threshold (default: 0.40)")
    parser.add_argument("--images", nargs="+", type=str,
                        help="Specific image paths to test")
    parser.add_argument("--max-per-class", type=int, default=2,
                        help="Max images per class for dataset sampling (default: 2)")
    args = parser.parse_args()

    # Load class names
    try:
        class_names = load_class_mapping()
    except Exception:
        logger.error("Could not load class_mapping.json")
        sys.exit(1)

    logger.info(f"Classes: {len(class_names)}")

    # Prepare samples
    if args.images:
        samples = [(Path(p), "unknown") for p in args.images]
    else:
        samples = get_sample_images("FruitGrade_Dataset", max_per_class=args.max_per_class)

    logger.info(f"Testing with {len(samples)} images")

    model_paths = {
        "v2": "ml/models/saved/model_v2.pth",
        "v3": "ml/models/saved/model_v3.pth",
    }

    if args.compare:
        results = []
        for version in ["v2", "v3"]:
            path = model_paths[version]
            if not os.path.exists(path):
                logger.warning(f"Model {version} not found at {path}, skipping")
                continue
            r = test_model(path, version, class_names, args.threshold, samples)
            results.append(r)

        if len(results) == 2:
            logger.info(f"\n{'='*60}")
            logger.info("COMPARISON SUMMARY")
            logger.info(f"{'='*60}")
            for r in results:
                logger.info(
                    f"  {r['model']:4s} | "
                    f"Accuracy: {r['accuracy']:.1%} | "
                    f"Fruit: {r['fruit_accuracy']:.1%} | "
                    f"Unknown: {r['unknown_rate']:.1%} | "
                    f"Avg Conf: {r['avg_conf']:.4f}"
                )
            logger.info(f"{'='*60}")
    else:
        path = model_paths[args.model]
        if not os.path.exists(path):
            logger.error(f"Model {args.model} not found at {path}")
            sys.exit(1)
        test_model(path, args.model, class_names, args.threshold, samples)


if __name__ == "__main__":
    main()
