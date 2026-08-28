"""
For filenames that were reported as fully missing (not just an extension
mismatch), search the folder for anything sharing the same leading number
in case they were extracted/renamed slightly differently
(e.g. "59 (3).jpg" with a space, or a different suffix numbering).

Usage:
    python find_near_matches.py
"""
import os
import re

IMAGE_DIR = os.path.join("data", "raw", "insat_kaggle", "insat3d_raw_cyclone_ds",
                          "CYCLONE_DATASET_FINAL")

TRULY_MISSING = ["34(1).jpg", "59(3).jpg", "81(1).jpg", "84(1).jpg"]

actual_files = sorted(os.listdir(IMAGE_DIR))

for target in TRULY_MISSING:
    m = re.match(r"(\d+)", target)
    number = m.group(1) if m else None
    print(f"\nLooking for files starting with '{number}' (target was '{target}'):")
    matches = [f for f in actual_files if re.match(rf"^{number}\D", f) or f == f"{number}.jpg"
               or f.startswith(f"{number}(") or f.startswith(f"{number} (")]
    if matches:
        for m2 in matches:
            print(f"   found: {m2}")
    else:
        print("   -> nothing found with this number at all")
