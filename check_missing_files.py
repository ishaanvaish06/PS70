"""
Quick diagnostic: find exactly which filenames from the CSV are missing
from the image folder, and check if it's just an extension mismatch
(.jpg vs .jpeg) or something else entirely.

Usage:
    python check_missing_files.py
"""
import os
import pandas as pd

CSV_PATH = os.path.join("data", "raw", "insat_kaggle", "insat_3d_ds - Sheet.csv")
IMAGE_DIR = os.path.join("data", "raw", "insat_kaggle", "insat3d_raw_cyclone_ds",
                          "CYCLONE_DATASET_FINAL")

df = pd.read_csv(CSV_PATH)
df.columns = [c.strip().lower() for c in df.columns]

actual_files = set(os.listdir(IMAGE_DIR))
# Build a lowercase lookup too, in case of case-sensitivity differences
actual_files_lower = {f.lower(): f for f in actual_files}

print(f"Checking {len(df)} filenames from CSV against {len(actual_files)} files in folder...\n")

missing = []
for fname in df["img_name"]:
    if fname in actual_files:
        continue
    missing.append(fname)

for fname in missing:
    base, ext = os.path.splitext(fname)
    print(f"MISSING: '{fname}'")

    # Check case-insensitive match
    if fname.lower() in actual_files_lower:
        print(f"   -> Found with different case: '{actual_files_lower[fname.lower()]}'")
        continue

    # Check alternate common extensions
    found_alt = False
    for alt_ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        alt_name = base + alt_ext
        if alt_name in actual_files:
            print(f"   -> Found with different extension: '{alt_name}'")
            found_alt = True
            break
    if not found_alt:
        print(f"   -> No similarly-named file found at all in the folder")

print(f"\nTotal missing: {len(missing)}")
