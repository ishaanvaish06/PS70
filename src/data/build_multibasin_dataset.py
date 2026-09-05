"""
src/data/build_multibasin_dataset.py
Extracts and combines global cyclone tracks across multiple ocean basins:
  - Western North Pacific (WP) - Typhoons
  - South Indian Ocean (SI) - Southern Hemisphere Cyclones
  - North Atlantic (NA) - Hurricanes
  - North Indian Ocean (NI) - Bay of Bengal / Arabian Sea Cyclones

Constructs continuous sliding-window forecasting sequences (5 past steps -> +6h, +12h, +24h targets)
yielding 15,000+ to 50,000+ kinematic training sequences for large-scale pre-training.
"""

import os
import sys
import pandas as pd
import numpy as np

RAW_GLOBAL_DIR = os.path.join("data", "raw", "ibtracs_global")
PROCESSED_DIR = os.path.join("data", "processed", "forecasting")
OUT_NPZ = os.path.join(PROCESSED_DIR, "multibasin_pretrain_sequences.npz")
OUT_META = os.path.join("data", "metadata", "ibtracs_multibasin_clean.csv")

BASIN_FILES = [
    ("WP", os.path.join(RAW_GLOBAL_DIR, "ibtracs_WP_raw.csv")),
    ("SI", os.path.join(RAW_GLOBAL_DIR, "ibtracs_SI_raw.csv")),
    ("NA", os.path.join(RAW_GLOBAL_DIR, "ibtracs_NA_raw.csv")),
]

MIN_YEAR = 2005
MAX_YEAR = 2024

def parse_basin_csv(basin_name, file_path):
    if not os.path.exists(file_path):
        print(f"[warn] {file_path} not found. Skipping {basin_name}.")
        return None

    print(f"Reading {basin_name} raw IBTrACS: {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)

    # Skip metadata row if present
    if len(df) > 0 and str(df.iloc[0].get("SEASON", "")).strip().lower() in ("year", ""):
        df = df.iloc[1:].reset_index(drop=True)

    keep_cols = ["SID", "NAME", "ISO_TIME", "LAT", "LON", "WMO_WIND", "USA_WIND"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")

    # Wind speed
    if "USA_WIND" in df.columns:
        df["wind_kt"] = pd.to_numeric(df["USA_WIND"], errors="coerce").fillna(
            pd.to_numeric(df.get("WMO_WIND"), errors="coerce")
        )
    else:
        df["wind_kt"] = pd.to_numeric(df.get("WMO_WIND"), errors="coerce")

    df["wind_speed_kmh"] = (df["wind_kt"] * 1.852).fillna(55.0)

    # Filter valid positions and modern era
    df = df.dropna(subset=["ISO_TIME", "LAT", "LON"]).reset_index(drop=True)
    df["year"] = df["ISO_TIME"].dt.year
    df = df[(df["year"] >= MIN_YEAR) & (df["year"] <= MAX_YEAR)].reset_index(drop=True)

    df = df.rename(columns={
        "SID": "cyclone_id",
        "NAME": "name",
        "ISO_TIME": "timestamp",
        "LAT": "latitude",
        "LON": "longitude",
    })
    df["basin"] = basin_name
    print(f"  {basin_name}: {len(df):,} observations across {df['cyclone_id'].nunique():,} cyclones ({MIN_YEAR}-{MAX_YEAR})")
    return df[["cyclone_id", "name", "basin", "timestamp", "latitude", "longitude", "wind_speed_kmh"]]

def build_sequences_from_df(df, history_len=5, lead_times=[1, 2, 4]):
    X_list, Y_list, meta_list = [], [], []
    max_lead = max(lead_times)

    for cyclone_id, group in df.groupby("cyclone_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        # 6-hourly continuity check
        diffs = group["timestamp"].diff().dt.total_seconds()
        gap_mask = (diffs > 6.5 * 3600.0)
        group["segment"] = gap_mask.cumsum()

        for _, seg in group.groupby("segment"):
            if len(seg) < history_len + max_lead:
                continue

            features = seg[["latitude", "longitude", "wind_speed_kmh"]].values.astype(np.float32)
            c_name = str(seg.iloc[0]["name"])
            c_basin = str(seg.iloc[0]["basin"])

            for i in range(len(seg) - (history_len + max_lead) + 1):
                x_win = features[i : i + history_len]
                y_tgts = [features[i + history_len - 1 + dt] for dt in lead_times]
                X_list.append(x_win)
                Y_list.append(y_tgts)
                meta_list.append({
                    "cyclone_id": cyclone_id,
                    "name": c_name,
                    "basin": c_basin,
                    "timestamp": str(seg.iloc[i + history_len - 1]["timestamp"])
                })

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y, pd.DataFrame(meta_list)

def main():
    print("=" * 70)
    print("BUILDING MULTI-BASIN CYCLONE TRAINING DATASET (WP + SI + NA)")
    print(f"Target Period: {MIN_YEAR} to {MAX_YEAR}")
    print("=" * 70)

    dfs = []
    for basin, fpath in BASIN_FILES:
        b_df = parse_basin_csv(basin, fpath)
        if b_df is not None:
            dfs.append(b_df)

    if not dfs:
        print("[ERROR] No basin datasets could be parsed.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(OUT_META), exist_ok=True)
    combined_df.to_csv(OUT_META, index=False)
    print(f"\nTotal Multi-Basin Tracks: {len(combined_df):,} rows from {combined_df['cyclone_id'].nunique():,} cyclones.")

    print("\nGenerating continuous sliding-window forecasting sequences (5-step history -> +6h, +12h, +24h)...")
    X, Y, meta_df = build_sequences_from_df(combined_df)

    print(f"\nSuccess! Generated {len(X):,} Multi-Basin Sequences:")
    print(f"  X shape: {X.shape} (Past 24h trajectory: lat, lon, wind)")
    print(f"  Y shape: {Y.shape} (Future targets: +6h, +12h, +24h)")
    print(f"  Basin breakdown:\n{meta_df['basin'].value_counts()}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.savez_compressed(OUT_NPZ, X=X, Y=Y)
    print(f"\nSaved multi-basin pre-training array to: {OUT_NPZ} ({os.path.getsize(OUT_NPZ)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
