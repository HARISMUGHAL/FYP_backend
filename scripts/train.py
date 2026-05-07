"""
scripts/train.py
----------------
Primary training script for the fruit classification model.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/config.yaml

This script uses the simple data loader from ml/preprocessing/dataset.py
which automatically scans FruitGrade_Dataset/<fruit>/<grade>/ folders.
"""

import os
import sys
import yaml

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.preprocessing.dataset import get_dataloaders
from ml.models.fruit_model import build_model
from ml.training.trainer import train_model
from ml.evaluation.evaluator import evaluate_model
from ml.utils.helpers import save_class_mapping

def main():
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # ── Flexible dataset path resolution ──────────────────────
    data_dir = (
        config.get("dataset", {}).get("path")
        or config.get("data", {}).get("raw_dir")
        or "FruitGrade_Dataset"
    )
    if not os.path.exists(data_dir):
        print(f"Error: Dataset path '{data_dir}' does not exist.")
        print("Please ensure the dataset is placed correctly.")
        return
        
    print("Loading dataset and creating dataloaders...")
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        img_size=config.get('dataset', {}).get('img_size', 224),
        batch_size=config.get('dataset', {}).get('batch_size', 32),
        val_split=config.get('dataset', {}).get('val_split', 0.2),
        num_workers=config.get('dataset', {}).get('num_workers', 0),
    )
    
    if len(class_names) == 0:
        print("No classes found in dataset. Please check the dataset structure.")
        return
        
    print(f"Found {len(class_names)} classes: {class_names}")
    save_class_mapping(class_names)
    
    print("Building model...")
    model = build_model(
        num_classes=len(class_names),
        pretrained=config.get('model', {}).get('pretrained', True),
        dropout_rate=config.get('model', {}).get('dropout_rate', 0.4),
    )
    
    print("Starting training...")
    best_model = train_model(model, train_loader, val_loader, config, class_names)
    
    print("Evaluating best model...")
    evaluate_model(best_model, val_loader, class_names)
    print(f"Training complete. Model saved to {config.get('model', {}).get('save_path', 'ml/models/saved/best_model.pth')}")

if __name__ == "__main__":
    main()
