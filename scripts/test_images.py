# ===== FIX PYTHON IMPORT PATH =====
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# ===== IMPORTS =====
from ml.pipeline.inference_pipeline import InferencePipeline
from ml.utils.helpers import load_yaml

# ===== LOAD CONFIG =====
cfg_path = os.path.join(BASE_DIR, "configs", "config.yaml")
cfg = load_yaml(cfg_path)

# ===== CHECK DATASET EXISTS =====
train_dir = os.path.join(BASE_DIR, cfg["data"]["train_dir"])
if not os.path.exists(train_dir):
    print(f"❌ Train folder not found: {train_dir}")
    print("👉 FIX: Run this first:")
    print("python scripts/train.py --skip-train")
    exit()

# ===== INITIALIZE PIPELINE =====
pipe = InferencePipeline(cfg)

# ===== PATHS =====
IMAGE_FOLDER = os.path.join(BASE_DIR, "test_images")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "test_outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===== RUN TESTING =====
print("\n🚀 Starting Testing...\n")

if not os.path.exists(IMAGE_FOLDER):
    print(f"❌ Image folder not found: {IMAGE_FOLDER}")
    exit()

for img_name in os.listdir(IMAGE_FOLDER):

    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):

        img_path = os.path.join(IMAGE_FOLDER, img_name)
        save_path = os.path.join(OUTPUT_FOLDER, f"out_{img_name}")

        print(f"📸 Processing: {img_name}")

        try:
            result = pipe.run_on_image(
                img_path,
                visualize=True,
                vis_save_path=save_path
            )

            print("✅ Result:")
            print(result)

        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")

print("\n🎯 Testing Complete!\n")