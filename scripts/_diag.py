"""Temporary diagnostic script — safe to delete after use."""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import torch
from torchvision import datasets

# --- checkpoint ---
ckpt_path = BASE_DIR / "ml" / "models" / "saved" / "best_model.pth"
ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
print("=== CHECKPOINT ===")
print("  epoch      :", ckpt.get("epoch"))
print("  metrics    :", ckpt.get("metrics"))
print("  num_classes:", ckpt.get("num_classes"))

# --- train split class names ---
train_root = BASE_DIR / "ml" / "data" / "train"
print("\n=== TRAIN SPLIT CLASSES ===")
exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
counts = {}
for cls_dir in sorted(train_root.iterdir()):
    if cls_dir.is_dir():
        c = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in exts)
        counts[cls_dir.name] = c

for label, cnt in sorted(counts.items()):
    print(f"  {label:<22} {cnt:>5}")
print(f"\n  Total classes : {len(counts)}")
print(f"  Total images  : {sum(counts.values())}")

# --- class ordering via ImageFolder (what model sees) ---
ds = datasets.ImageFolder(root=str(train_root))
print("\n=== ImageFolder class order (model label indices) ===")
for i, cls in enumerate(ds.classes):
    print(f"  {i:>2}: {cls}")
