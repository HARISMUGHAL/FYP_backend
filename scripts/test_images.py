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

# ===== INIT PIPELINE =====
pipe = InferencePipeline(cfg)

# ===== TEST IMAGE FOLDER =====
IMAGE_FOLDER = os.path.join(BASE_DIR, "test_images")

print("\n🚀 Starting Testing...\n")

if not os.path.exists(IMAGE_FOLDER):
    print(f"❌ Folder not found: {IMAGE_FOLDER}")
    exit()

# ===== LOOP THROUGH IMAGES =====
for img_name in os.listdir(IMAGE_FOLDER):

    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):

        img_path = os.path.join(IMAGE_FOLDER, img_name)

        print(f"\n📸 Image: {img_name}")

        try:
            result = pipe.run_on_image(img_path)

            # ===== CLEAN OUTPUT =====
            if isinstance(result, list) and len(result) > 0:
                res = result[0]
            else:
                res = result

            print(f"Fruit: {res.get('fruit')}")
            print(f"Grade: {res.get('grade')}")
            print(f"Confidence: {round(res.get('confidence', 0), 3)}")

        except Exception as e:
            print(f"❌ Error: {e}")

print("\n🎯 Testing Complete!\n")