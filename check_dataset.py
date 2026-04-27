from pathlib import Path

DATASET_PATH = Path("FruitGrade_Dataset")

FRUITS = [
    "apple","Apricot","banana","Grapes",
    "guava","Honeydew","Orange","Tangerine"
]

GRADES = ["A","B","C"]

# ✅ ALL IMAGE TYPES
EXTENSIONS = ["*.jpg","*.jpeg","*.png","*.webp","*.bmp"]

def count_images(folder):
    total = 0
    for ext in EXTENSIONS:
        total += len(list(folder.rglob(ext)))
    return total

print("="*50)
print("FINAL DATASET REPORT (ALL IMAGES)")
print("="*50)

total_images = 0
complete = True

for fruit in FRUITS:
    print(f"\n{fruit}:")

    fruit_path = DATASET_PATH / fruit
    fruit_total = 0

    for grade in GRADES:
        grade_count = 0

        # 🔥 deep search
        for sub in fruit_path.rglob("*"):
            if sub.is_dir() and sub.name.upper() == grade:
                grade_count += count_images(sub)

        fruit_total += grade_count

        status = "OK" if grade_count >= 600 else "MISSING"

        if grade_count < 600:
            complete = False

        print(f"  {grade}: {grade_count} → {status}")

    print(f"  Total: {fruit_total}")
    total_images += fruit_total

print("\n" + "="*50)
print(f"TOTAL IMAGES: {total_images}")
print("="*50)

if complete:
    print("\n✅ DATASET COMPLETE")
else:
    print("\n❌ DATASET INCOMPLETE")