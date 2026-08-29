"""
build_datasets.py
Data processing script for Person 1 (Data Engineer) — SIH 2026 PS 26070
Generates:
1. Standardized master_dataset.csv (IBTrACS + ERA5)
2. Cyclone-level train/val/test splits (70/15/15)
3. Dataset A (Detection/Presence)
4. Dataset B (Classification & Intensity: Multi-Source & Image-Only)
5. Dataset C (Forecasting Sequences: past 24h -> +6h, +12h, +24h)
"""

import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def main():
    print("=" * 70)
    print("PERSON 1 — BUILDING ML-READY DATASETS & SPLITS")
    print("=" * 70)

    os.makedirs("data/processed/detection", exist_ok=True)
    os.makedirs("data/processed/classification", exist_ok=True)
    os.makedirs("data/processed/forecasting", exist_ok=True)
    os.makedirs("data/metadata", exist_ok=True)

    # -------------------------------------------------------------
    # 1. Standardize Master Dataset
    # -------------------------------------------------------------
    print("\n[1/5] Standardizing Master Dataset...")
    src_path = "data/metadata/ibtracs_with_era5.csv"
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file {src_path} not found!")

    df = pd.read_csv(src_path)
    print(f"Loaded {len(df)} rows from {src_path}")

    # Standardize column names to match integration contract
    # Contract: cyclone_id, season, name, subbasin, timestamp, lat, lon, wind_speed, pressure, category, sst, wind_u, wind_v
    rename_dict = {
        "latitude": "lat",
        "longitude": "lon",
        "wind_speed_kmh": "wind_speed",
        "pressure_hpa": "pressure",
        "u_wind": "wind_u",
        "v_wind": "wind_v",
        "sst_celsius": "sst",
        "pressure_msl_hpa": "pressure_msl"
    }
    df = df.rename(columns=rename_dict)
    
    # Ensure timestamp is ISO format
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["cyclone_id", "timestamp"]).reset_index(drop=True)

    # Add pre-genesis pattern flag (Page 1 spec alignment):
    # Favorable pre-genesis conditions: SST >= 26.5°C with Depression stage or early formation
    df["pre_genesis_favorable"] = (
        (df["sst"] >= 26.5) & 
        (df["category"].fillna("Depression").isin(["Depression", "Deep Depression"]))
    ).astype(bool)

    # Select and order columns
    cols = [
        "cyclone_id", "season", "name", "subbasin", "timestamp",
        "lat", "lon", "wind_speed", "pressure", "category",
        "sst", "wind_u", "wind_v", "pressure_msl", "pre_genesis_favorable"
    ]
    # Keep only available columns
    cols = [c for c in cols if c in df.columns]
    master_df = df[cols].copy()

    master_path_proc = "data/processed/master_dataset.csv"
    master_path_meta = "data/metadata/master_dataset.csv"
    master_df.to_csv(master_path_proc, index=False)
    master_df.to_csv(master_path_meta, index=False)
    print(f" Saved Master Dataset: {len(master_df)} rows, {len(master_df.columns)} columns")
    print(f"  -> {master_path_proc}")
    print(f"  -> {master_path_meta}")

    # -------------------------------------------------------------
    # 2. Cyclone-Level Train / Val / Test Split
    # -------------------------------------------------------------
    print("\n[2/5] Creating Cyclone-Level Train/Validation/Test Splits (70/15/15)...")
    cyclone_ids = master_df["cyclone_id"].unique()
    num_cyclones = len(cyclone_ids)
    print(f"Total unique cyclones: {num_cyclones}")

    # Deterministic split based on sorted cyclone IDs with a fixed seed
    np.random.seed(42)
    shuffled_cyclones = np.random.permutation(cyclone_ids)

    n_train = int(0.70 * num_cyclones)
    n_val = int(0.15 * num_cyclones)

    train_cids = set(shuffled_cyclones[:n_train])
    val_cids = set(shuffled_cyclones[n_train:n_train + n_val])
    test_cids = set(shuffled_cyclones[n_train + n_val:])

    print(f"  Train cyclones: {len(train_cids)} ({len(train_cids)/num_cyclones*100:.1f}%)")
    print(f"  Val cyclones:   {len(val_cids)} ({len(val_cids)/num_cyclones*100:.1f}%)")
    print(f"  Test cyclones:  {len(test_cids)} ({len(test_cids)/num_cyclones*100:.1f}%)")

    # Save cyclone ID lists
    pd.DataFrame({"cyclone_id": sorted(list(train_cids))}).to_csv("data/metadata/train_cyclones.csv", index=False)
    pd.DataFrame({"cyclone_id": sorted(list(val_cids))}).to_csv("data/metadata/validation_cyclones.csv", index=False)
    pd.DataFrame({"cyclone_id": sorted(list(test_cids))}).to_csv("data/metadata/test_cyclones.csv", index=False)

    # Split master dataset
    train_master = master_df[master_df["cyclone_id"].isin(train_cids)].reset_index(drop=True)
    val_master = master_df[master_df["cyclone_id"].isin(val_cids)].reset_index(drop=True)
    test_master = master_df[master_df["cyclone_id"].isin(test_cids)].reset_index(drop=True)

    train_master.to_csv("data/metadata/train.csv", index=False)
    val_master.to_csv("data/metadata/validation.csv", index=False)
    test_master.to_csv("data/metadata/test.csv", index=False)

    print(f"  Split rows -> Train: {len(train_master)}, Val: {len(val_master)}, Test: {len(test_master)}")

    # -------------------------------------------------------------
    # 3. Build Dataset C: Forecasting Sequences
    # -------------------------------------------------------------
    print("\n[3/5] Building Dataset C (Forecasting Sequences for Person 4)...")
    # Sequence definition:
    # History: t-24h, t-18h, t-12h, t-6h, t (5 steps, 6h intervals)
    # Forecast targets: t+6h, t+12h, t+24h (3 lead times)
    # Input features: [lat, lon, wind_speed, pressure, sst, wind_u, wind_v] (7 features)
    # Target features: [lat, lon, wind_speed] (3 features per forecast step)

    feature_cols = ["lat", "lon", "wind_speed", "pressure", "sst", "wind_u", "wind_v"]
    target_cols = ["lat", "lon", "wind_speed"]

    def extract_sequences(df_subset, split_name):
        X_list = []
        Y_list = []
        meta_list = []

        for cid, group in df_subset.groupby("cyclone_id"):
            grp = group.sort_values("timestamp").copy()
            # Must have at least a few points
            if len(grp) < 3:
                continue
                
            grp = grp.set_index("timestamp")
            
            # Fill missing numerical values forward/backward within storm
            grp_interp = grp[feature_cols].interpolate(method="time").ffill().bfill()
            
            times = grp_interp.index
            
            # Slide over valid timestamps
            for t_curr in times:
                t_hist = [t_curr - timedelta(hours=h) for h in [24, 18, 12, 6, 0]]
                t_future = [t_curr + timedelta(hours=h) for h in [6, 12, 24]]

                # Check bounds
                if t_hist[0] < times[0] or t_future[-1] > times[-1]:
                    continue
                
                try:
                    # Sample features at exact times via reindex & time interpolation
                    combined_idx = grp_interp.index.union(t_hist).union(t_future)
                    interp_full = grp_interp.reindex(combined_idx).interpolate(method="time").ffill().bfill()
                    
                    hist_sample = interp_full.loc[t_hist]
                    fut_sample = interp_full.loc[t_future]

                    x_arr = hist_sample[feature_cols].values # shape (5, 7)
                    y_arr = fut_sample[target_cols].values   # shape (3, 3)

                    if np.isnan(x_arr).any() or np.isnan(y_arr).any():
                        continue

                    X_list.append(x_arr)
                    Y_list.append(y_arr)
                    meta_list.append({
                        "cyclone_id": cid,
                        "t_zero": str(t_curr),
                        "t_minus_24h": str(t_hist[0]),
                        "t_plus_24h": str(t_future[-1]),
                        "origin_lat": x_arr[-1, 0],
                        "origin_lon": x_arr[-1, 1],
                        "origin_wind": x_arr[-1, 2],
                        "target_6h_lat": y_arr[0, 0],
                        "target_6h_lon": y_arr[0, 1],
                        "target_6h_wind": y_arr[0, 2],
                        "target_12h_lat": y_arr[1, 0],
                        "target_12h_lon": y_arr[1, 1],
                        "target_12h_wind": y_arr[1, 2],
                        "target_24h_lat": y_arr[2, 0],
                        "target_24h_lon": y_arr[2, 1],
                        "target_24h_wind": y_arr[2, 2],
                    })
                except Exception as e:
                    continue

        X = np.array(X_list, dtype=np.float32) if len(X_list) > 0 else np.empty((0, 5, 7), dtype=np.float32)
        Y = np.array(Y_list, dtype=np.float32) if len(Y_list) > 0 else np.empty((0, 3, 3), dtype=np.float32)
        meta_df = pd.DataFrame(meta_list)
        return X, Y, meta_df

    X_train, Y_train, meta_train = extract_sequences(train_master, "train")
    X_val, Y_val, meta_val = extract_sequences(val_master, "val")
    X_test, Y_test, meta_test = extract_sequences(test_master, "test")

    print(f"  Forecasting Sequences -> Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"  Forecasting Sequences -> Val:   X={X_val.shape}, Y={Y_val.shape}")
    print(f"  Forecasting Sequences -> Test:  X={X_test.shape}, Y={Y_test.shape}")

    np.savez_compressed("data/processed/forecasting/train_sequences.npz", X=X_train, Y=Y_train, features=feature_cols, targets=target_cols)
    np.savez_compressed("data/processed/forecasting/val_sequences.npz", X=X_val, Y=Y_val, features=feature_cols, targets=target_cols)
    np.savez_compressed("data/processed/forecasting/test_sequences.npz", X=X_test, Y=Y_test, features=feature_cols, targets=target_cols)

    meta_train.to_csv("data/processed/forecasting/train_sequences_metadata.csv", index=False)
    meta_val.to_csv("data/processed/forecasting/val_sequences_metadata.csv", index=False)
    meta_test.to_csv("data/processed/forecasting/test_sequences_metadata.csv", index=False)

    # -------------------------------------------------------------
    # 4. Build Dataset B: Classification & Intensity
    # -------------------------------------------------------------
    print("\n[4/5] Building Dataset B (Classification & Intensity for Person 3)...")
    class_cols = ["cyclone_id", "season", "name", "subbasin", "timestamp", "lat", "lon", "sst", "pressure_msl", "wind_u", "wind_v", "wind_speed", "pressure", "category", "pre_genesis_favorable"]
    
    def build_multisource_class(df_subset, split_name):
        valid = df_subset[df_subset["category"].notna() & df_subset["wind_speed"].notna()].copy()
        valid = valid[[c for c in class_cols if c in valid.columns]]
        out_path = f"data/processed/classification/multisource_{split_name}.csv"
        valid.to_csv(out_path, index=False)
        return len(valid)

    n_ms_train = build_multisource_class(train_master, "train")
    n_ms_val = build_multisource_class(val_master, "val")
    n_ms_test = build_multisource_class(test_master, "test")
    print(f"  Multi-source Classification -> Train: {n_ms_train} rows, Val: {n_ms_val} rows, Test: {n_ms_test} rows")

    # Image-only Kaggle dataset splits
    kaggle_labels_path = "data/processed/classification/image_only_kaggle/labels.csv"
    if os.path.exists(kaggle_labels_path):
        kdf = pd.read_csv(kaggle_labels_path)
        np.random.seed(42)
        shuffled_idx = np.random.permutation(len(kdf))
        k_tr = int(0.70 * len(kdf))
        k_v = int(0.15 * len(kdf))
        
        tr_k = kdf.iloc[shuffled_idx[:k_tr]].reset_index(drop=True)
        v_k = kdf.iloc[shuffled_idx[k_tr:k_tr + k_v]].reset_index(drop=True)
        te_k = kdf.iloc[shuffled_idx[k_tr + k_v:]].reset_index(drop=True)

        tr_k.to_csv("data/processed/classification/image_only_kaggle/train_labels.csv", index=False)
        v_k.to_csv("data/processed/classification/image_only_kaggle/val_labels.csv", index=False)
        te_k.to_csv("data/processed/classification/image_only_kaggle/test_labels.csv", index=False)
        print(f"  Image-Only Kaggle -> Train: {len(tr_k)}, Val: {len(v_k)}, Test: {len(te_k)}")

        # -------------------------------------------------------------
        # 5. Build Dataset A: Detection & Presence
        # -------------------------------------------------------------
        print("\n[5/5] Building Dataset A (Detection / Presence for Person 2)...")
        det_df = kdf.copy()
        det_df["cyclone_detected"] = True
        det_df["image_path"] = "data/processed/classification/image_only_kaggle/images/" + det_df["filename"]
        # Add mock bbox for contract schema compatibility
        det_df["mock_bbox"] = "[420, 190, 600, 370]"
        det_df["structural_pattern"] = det_df["category"].apply(
            lambda c: "eye_visible" if "Severe" in str(c) else "curved_band" if "Cyclonic" in str(c) else "shear_pattern"
        )
        
        det_df.iloc[shuffled_idx[:k_tr]].to_csv("data/processed/detection/train_detection.csv", index=False)
        det_df.iloc[shuffled_idx[k_tr:k_tr + k_v]].to_csv("data/processed/detection/val_detection.csv", index=False)
        det_df.iloc[shuffled_idx[k_tr + k_v:]].to_csv("data/processed/detection/test_detection.csv", index=False)
        det_df.to_csv("data/processed/detection/detection_all.csv", index=False)
        print(f"  Detection/Presence -> Saved {len(det_df)} records to data/processed/detection/")

    print("\n" + "=" * 70)
    print("ALL DATASETS AND SPLITS SUCCESSFULLY BUILT!")
    print("=" * 70)

if __name__ == "__main__":
    main()
