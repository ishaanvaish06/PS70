"""
Step 2 (Track A) — Person 1 (Data Engineer)
Clean the Kaggle INSAT-3D intensity dataset (img_name -> knots) into a
standalone image-only classification/intensity dataset for Person 3.

IMPORTANT: This dataset has NO timestamp, lat/lon, or cyclone identity.
It CANNOT be synced into master_dataset.csv (which needs IBTrACS + ERA5
alignment). It is handed off separately as an image-only baseline dataset,
which is exactly what Person 3's "image-only vs multi-source" comparison
experiment needs.

Usage:
    python clean_kaggle_intensity.py

Input (expected, adjust CSV_PATH/IMAGE_DIR if yours differ):
    data/raw/insat_kaggle/insat_3d_ds - Sheet.csv
    data/raw/insat_kaggle/insat3d_raw_cyclone_ds/CYCLONE_DATASET_FINAL/*.jpg

Output:
    data/processed/classification/image_only_kaggle/labels.csv
    data/processed/classification/image_only_kaggle/images/<file>   (copied)
"""

import os
import shutil
import pandas as pd

CSV_PATH = os.path.join("data", "raw", "insat_kaggle", "insat_3d_ds - Sheet.csv")
IMAGE_DIR = os.path.join("data", "raw", "insat_kaggle", "insat3d_raw_cyclone_ds",
                          "CYCLONE_DATASET_FINAL")

OUT_DIR = os.path.join("data", "processed", "classification", "image_only_kaggle")
OUT_IMAGES_DIR = os.path.join(OUT_DIR, "images")
OUT_LABELS_PATH = os.path.join(OUT_DIR, "labels.csv")


def classify_imd_category(wind_kmh):
    """Same IMD/RSMC New Delhi scale used in get_ibtracs.py, kept identical
    on purpose so Person 3's categories are consistent across both datasets."""
    if pd.isna(wind_kmh):
        return None
    if wind_kmh < 31:
        return "Low Pressure Area"
    elif wind_kmh < 50:
        return "Depression"
    elif wind_kmh < 62:
        return "Deep Depression"
    elif wind_kmh < 89:
        return "Cyclonic Storm"
    elif wind_kmh < 118:
        return "Severe Cyclonic Storm"
    elif wind_kmh < 166:
        return "Very Severe Cyclonic Storm"
    elif wind_kmh < 222:
        return "Extremely Severe Cyclonic Storm"
    else:
        return "Super Cyclonic Storm"


def clean():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"{CSV_PATH} not found. Update CSV_PATH at the top of this script "
            f"to match your actual file location/name.")
    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(
            f"{IMAGE_DIR} not found. Update IMAGE_DIR at the top of this script "
            f"to match your actual folder location.")

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df):,} rows from {CSV_PATH}")
    print(f"Columns: {list(df.columns)}")

    # Standardize column names (seen as img_name/label in the screenshot)
    df.columns = [c.strip().lower() for c in df.columns]
    if "img_name" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Expected columns 'img_name' and 'label', got {list(df.columns)}. "
            f"Update this script's column handling to match.")

    df = df.rename(columns={"img_name": "filename", "label": "wind_speed_kt"})
    df["wind_speed_kt"] = pd.to_numeric(df["wind_speed_kt"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["wind_speed_kt"]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with non-numeric/missing label")

    # Convert knots -> km/h (same convention as get_ibtracs.py) and assign
    # the IMD category so this dataset's labels are directly comparable to
    # the ones Person 3 trains against from master_dataset.csv.
    df["wind_speed_kmh"] = (df["wind_speed_kt"] * 1.852).round(1)
    df["category"] = df["wind_speed_kmh"].apply(classify_imd_category)

    # Check every listed image actually exists, and copy it into the
    # processed output folder. If the exact filename isn't found, try
    # common alternate extensions (.JPEG/.jpeg/.PNG etc.), and a small
    # set of known typo fixes seen in this specific dataset, before
    # giving up. Ambiguous cases (multiple possible source files, e.g.
    # duplicates renamed with a storm name instead of a number) are
    # deliberately NOT guessed — a wrong image/label pairing is worse
    # than a dropped row.
    os.makedirs(OUT_IMAGES_DIR, exist_ok=True)
    ALT_EXTS = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    # Confirmed unambiguous typo fixes (verified manually against the
    # actual folder listing) — extend this dict if you spot more.
    KNOWN_TYPO_FIXES = {
        "34(1).jpg": "34(!).jpg",
    }
    found_flags = []
    resolved_filenames = []
    for fname in df["filename"]:
        src = os.path.join(IMAGE_DIR, fname)
        actual_src = None
        if os.path.exists(src):
            actual_src = src
        elif fname in KNOWN_TYPO_FIXES:
            candidate = os.path.join(IMAGE_DIR, KNOWN_TYPO_FIXES[fname])
            if os.path.exists(candidate):
                actual_src = candidate
        else:
            base, _ = os.path.splitext(fname)
            for ext in ALT_EXTS:
                candidate = os.path.join(IMAGE_DIR, base + ext)
                if os.path.exists(candidate):
                    actual_src = candidate
                    break

        if actual_src:
            actual_fname = os.path.basename(actual_src)
            shutil.copy2(actual_src, os.path.join(OUT_IMAGES_DIR, actual_fname))
            found_flags.append(True)
            resolved_filenames.append(actual_fname)
        else:
            found_flags.append(False)
            resolved_filenames.append(fname)
    df["image_found"] = found_flags
    df["filename"] = resolved_filenames

    missing = (~df["image_found"]).sum()
    if missing:
        print(f"[warn] {missing} rows reference an image file that was not found "
              f"in {IMAGE_DIR} — check filename/extension mismatches (e.g. .jpg vs .jpeg)")

    df_final = df[df["image_found"]].drop(columns=["image_found"]).reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    df_final.to_csv(OUT_LABELS_PATH, index=False)

    print(f"\nSaved {len(df_final):,} labeled images to: {OUT_DIR}")
    print(f"Labels file: {OUT_LABELS_PATH}")
    print("\nCategory distribution:")
    print(df_final["category"].value_counts())
    print(f"\nWind speed range: {df_final['wind_speed_kmh'].min()} - "
          f"{df_final['wind_speed_kmh'].max()} km/h")

    return df_final


if __name__ == "__main__":
    clean()
