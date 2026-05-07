"""
scripts/_dataset_audit.py
-------------------------
READ-ONLY dataset audit script.
Does NOT modify any files.
Reports: image counts, corrupt images, duplicates, class balance.
"""

import os
import sys
import hashlib
from pathlib import Path
from PIL import Image

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "FruitGrade_Dataset"

def get_file_hash(filepath, chunk_size=8192):
    """Compute MD5 hash of a file for duplicate detection."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 70)
    print("DATASET AUDIT -- READ ONLY (no files modified)")
    print(f"Dataset path: {DATASET_DIR}")
    print("=" * 70)

    if not DATASET_DIR.exists():
        print(f"ERROR: Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)

    class_counts = {}
    corrupted = []
    small_files = []
    all_hashes = {}  # hash -> list of paths
    total_images = 0

    for fruit_name in sorted(os.listdir(DATASET_DIR)):
        fruit_path = DATASET_DIR / fruit_name
        if not fruit_path.is_dir():
            continue
        for grade in sorted(os.listdir(fruit_path)):
            grade_path = fruit_path / grade
            if not grade_path.is_dir():
                continue

            label = f"{fruit_name}_{grade}"
            count = 0

            for fname in os.listdir(grade_path):
                fpath = grade_path / fname
                if not fpath.is_file():
                    continue
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                count += 1
                total_images += 1

                # Check file size
                fsize = fpath.stat().st_size
                if fsize < 1024:
                    small_files.append((str(fpath), fsize))

                # Check for corruption
                try:
                    img = Image.open(fpath)
                    img.verify()
                except Exception as e:
                    corrupted.append((str(fpath), str(e)))

                # Duplicate detection
                try:
                    fhash = get_file_hash(fpath)
                    if fhash not in all_hashes:
                        all_hashes[fhash] = []
                    all_hashes[fhash].append(str(fpath))
                except Exception:
                    pass

            class_counts[label] = count

    # -- Report ---------------------------------------------
    print("\n" + "=" * 70)
    print("1. IMAGE COUNTS PER CLASS")
    print("=" * 70)
    counts_list = sorted(class_counts.items())
    max_count = max(class_counts.values()) if class_counts else 1
    min_count = min(class_counts.values()) if class_counts else 0
    avg_count = sum(class_counts.values()) / len(class_counts) if class_counts else 0

    for label, count in counts_list:
        bar = "#" * int(40 * count / max_count) if max_count > 0 else ""
        flag = " [!LOW]" if count < avg_count * 0.5 else ""
        print(f"  {label:25s}  {count:5d}  {bar}{flag}")

    print(f"\n  {'TOTAL':25s}  {total_images:5d}")
    print(f"  {'CLASSES':25s}  {len(class_counts):5d}")
    print(f"  {'AVG PER CLASS':25s}  {avg_count:7.1f}")
    print(f"  {'MIN':25s}  {min_count:5d}")
    print(f"  {'MAX':25s}  {max_count:5d}")
    print(f"  {'IMBALANCE RATIO':25s}  {max_count/min_count:.2f}x" if min_count > 0 else "")

    # ── Updated classes focus ──────────────────────────────
    print("\n" + "=" * 70)
    print("2. FOCUS: UPDATED CLASSES (apple & banana)")
    print("=" * 70)
    focus_classes = ["apple_A", "apple_B", "apple_C", "banana_A", "banana_B", "banana_C"]
    for cls in focus_classes:
        count = class_counts.get(cls, 0)
        print(f"  {cls:25s}  {count:5d}")

    # ── Confusion-risk classes ─────────────────────────────
    print("\n" + "=" * 70)
    print("3. CONFUSION-RISK CLASSES (apple vs guava)")
    print("=" * 70)
    confusion_classes = ["apple_A", "apple_B", "apple_C", "guava_A", "guava_B", "guava_C"]
    for cls in confusion_classes:
        count = class_counts.get(cls, 0)
        print(f"  {cls:25s}  {count:5d}")

    # ── Corrupted images ───────────────────────────────────
    print("\n" + "=" * 70)
    print("4. CORRUPTED IMAGES")
    print("=" * 70)
    if corrupted:
        for path, err in corrupted:
            print(f"  ❌ {path}")
            print(f"     Error: {err}")
    else:
        print("  ✅ No corrupted images found")

    # ── Small files ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("5. SUSPICIOUSLY SMALL FILES (< 1KB)")
    print("=" * 70)
    if small_files:
        for path, size in small_files:
            print(f"  ⚠ {path} ({size} bytes)")
    else:
        print("  ✅ No suspiciously small files")

    # ── Duplicates ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("6. DUPLICATE IMAGES")
    print("=" * 70)
    duplicates = {h: paths for h, paths in all_hashes.items() if len(paths) > 1}
    if duplicates:
        dup_count = sum(len(paths) - 1 for paths in duplicates.values())
        print(f"  ⚠ Found {dup_count} duplicate images in {len(duplicates)} groups:")
        for i, (h, paths) in enumerate(duplicates.items()):
            if i >= 20:  # Limit output
                print(f"  ... and {len(duplicates) - 20} more duplicate groups")
                break
            print(f"\n  Group {i+1} ({len(paths)} copies):")
            for p in paths:
                print(f"    - {p}")
    else:
        print("  ✅ No duplicate images found")

    # ── Weak classes & imbalance risks ─────────────────────
    print("\n" + "=" * 70)
    print("7. WEAK CLASSES & OVERFITTING RISKS")
    print("=" * 70)
    for label, count in sorted(class_counts.items(), key=lambda x: x[1]):
        if count < avg_count * 0.5:
            print(f"  ⚠ {label:25s}  {count:5d}  — UNDERREPRESENTED (< 50% of avg)")
    for label, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        if count > avg_count * 2:
            print(f"  ⚠ {label:25s}  {count:5d}  — OVERREPRESENTED (> 200% of avg)")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE — No files were modified")
    print("=" * 70)


if __name__ == "__main__":
    main()
