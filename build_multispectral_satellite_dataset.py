"""
build_multispectral_satellite_dataset.py
Pairs INSAT-3D Thermal Infrared (IR) and Visible (VIS) satellite imagery
into a unified multi-sensor satellite dataset for multi-source intensity classification
and cyclone pattern identification.

Output:
  data/processed/classification/multisource_satellite/
    - images_ir/
    - images_vis/
    - pairs_manifest.csv
    - train_pairs.csv, val_pairs.csv, test_pairs.csv
"""

import os
import shutil
import pandas as pd
import numpy as np

IR_DIR = os.path.join("data", "raw", "insat", "insat3d_ir_cyclone_ds", "CYCLONE_DATASET_INFRARED")
VIS_DIR = os.path.join("data", "raw", "insat", "insat3d_raw_cyclone_ds", "CYCLONE_DATASET_FINAL")
LABEL_CSV = os.path.join("data", "raw", "insat", "insat_3d_ds - Sheet.csv")

OUT_DIR = os.path.join("data", "processed", "classification", "multisource_satellite")
OUT_IR = os.path.join(OUT_DIR, "images_ir")
OUT_VIS = os.path.join(OUT_DIR, "images_vis")

IMD_CATEGORIES = [
    (0, 31, "Depression", "D"),
    (31, 50, "Depression", "D"),
    (50, 62, "Deep Depression", "DD"),
    (62, 89, "Cyclonic Storm", "CS"),
    (89, 118, "Severe Cyclonic Storm", "SCS"),
    (118, 167, "Very Severe Cyclonic Storm", "VSCS"),
    (167, 222, "Extremely Severe Cyclonic Storm", "ESCS"),
    (222, 999, "Super Cyclonic Storm", "SuCS")
]

CLASS_TO_IDX = {
    "Depression": 0,
    "Deep Depression": 1,
    "Cyclonic Storm": 2,
    "Severe Cyclonic Storm": 3,
    "Very Severe Cyclonic Storm": 4,
    "Extremely Severe Cyclonic Storm": 5,
    "Super Cyclonic Storm": 6
}

def get_imd_category(wind_kmh):
    for low, high, name, code in IMD_CATEGORIES:
        if low <= wind_kmh < high:
            return name, code
    return "Super Cyclonic Storm", "SuCS"

def main():
    print("=" * 60)
    print("BUILDING MULTI-SOURCE SATELLITE DATASET (IR + VISIBLE)")
    print("=" * 60)

    os.makedirs(OUT_IR, exist_ok=True)
    os.makedirs(OUT_VIS, exist_ok=True)

    labels_df = pd.read_csv(LABEL_CSV)
    labels_df.columns = [c.strip() for c in labels_df.columns]
    
    # Create filename -> wind_kt mapping
    label_map = {}
    for _, row in labels_df.iterrows():
        fname = str(row["img_name"]).strip()
        try:
            kt = float(row["label"])
            label_map[fname] = kt
        except (ValueError, TypeError):
            continue

    ir_files = set(os.listdir(IR_DIR))
    vis_files = set(os.listdir(VIS_DIR))
    common_files = sorted(list(ir_files.intersection(vis_files)))

    records = []
    for fname in common_files:
        if fname not in label_map:
            continue
        
        wind_kt = label_map[fname]
        wind_kmh = round(wind_kt * 1.852, 1)
        cat_name, imd_code = get_imd_category(wind_kmh)
        cat_idx = CLASS_TO_IDX[cat_name]

        # Copy to standardized directory
        src_ir = os.path.join(IR_DIR, fname)
        src_vis = os.path.join(VIS_DIR, fname)
        dst_ir = os.path.join(OUT_IR, fname)
        dst_vis = os.path.join(OUT_VIS, fname)

        shutil.copy2(src_ir, dst_ir)
        shutil.copy2(src_vis, dst_vis)

        records.append({
            "filename": fname,
            "path_ir": f"data/processed/classification/multisource_satellite/images_ir/{fname}",
            "path_vis": f"data/processed/classification/multisource_satellite/images_vis/{fname}",
            "wind_speed_kt": wind_kt,
            "wind_speed_kmh": wind_kmh,
            "category": cat_name,
            "imd_code": imd_code,
            "category_idx": cat_idx
        })

    df = pd.DataFrame(records)
    print(f"Matched & copied {len(df)} dual-sensor satellite pairs.")

    # Stratified split: 70% Train, 15% Val, 15% Test
    np.random.seed(42)
    shuffled_idx = np.random.permutation(len(df))
    n_train = int(0.70 * len(df))
    n_val = int(0.15 * len(df))

    train_df = df.iloc[shuffled_idx[:n_train]].reset_index(drop=True)
    val_df = df.iloc[shuffled_idx[n_train:n_train + n_val]].reset_index(drop=True)
    test_df = df.iloc[shuffled_idx[n_train + n_val:]].reset_index(drop=True)

    df.to_csv(os.path.join(OUT_DIR, "pairs_manifest.csv"), index=False)
    train_df.to_csv(os.path.join(OUT_DIR, "train_pairs.csv"), index=False)
    val_df.to_csv(os.path.join(OUT_DIR, "val_pairs.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test_pairs.csv"), index=False)

    print(f"Dataset partitions saved:")
    print(f"  Train pairs: {len(train_df)}")
    print(f"  Val pairs:   {len(val_df)}")
    print(f"  Test pairs:  {len(test_df)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
