"""
Diagnostic: show a random sample of what's actually inside each dataset subfolder.
Helps confirm images are correctly labeled visually.
"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
import random
from PIL import Image

dataset = BASE_DIR / "FruitGrade_Dataset"
exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

print("=== IMAGE SIZE SAMPLE PER CLASS ===")
for fruit_dir in sorted(dataset.iterdir()):
    if not fruit_dir.is_dir():
        continue
    for grade_dir in sorted(fruit_dir.iterdir()):
        if not grade_dir.is_dir() or grade_dir.name.upper() not in ("A","B","C"):
            continue
        all_imgs = [f for f in grade_dir.rglob("*") if f.suffix.lower() in exts]
        if not all_imgs:
            continue
        sample = random.choice(all_imgs)
        try:
            with Image.open(sample) as im:
                w, h = im.size
                mode = im.mode
        except Exception as e:
            w, h, mode = "?","?",str(e)
        label = f"{fruit_dir.name}_{grade_dir.name.upper()}"
        print(f"  {label:<20}  count={len(all_imgs):>4}  sample_size={w}x{h}  mode={mode}")
