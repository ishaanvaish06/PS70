"""
build_satellite_sequences.py
Builds spatiotemporal satellite video sequences:
  X_video: (N, T=5, C=2, H=128, W=128) - 5 consecutive time-series satellite frames (Thermal IR + VIS)
  X_steering: (N, T=5, F=5) - 5 timesteps of 500 hPa & 700 hPa steering flow (z_500, u_500, v_500, u_700, v_700)
  Y_target: (N, 3, 3) - (+6h, +12h, +24h) lat, lon, wind_speed
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

OUT_DIR = os.path.join("data", "processed", "forecasting", "multimodal_sequences")
STEERING_CSV = os.path.join("data", "metadata", "ibtracs_with_steering.csv")

def build_multimodal_dataset(history_len=5, lead_times=[1, 2, 4], img_size=(128, 128)):
    print("=" * 70)
    print("BUILDING MULTIMODAL SPATIOTEMPORAL SATELLITE + 500hPa STEERING SEQUENCES")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(STEERING_CSV):
        raise FileNotFoundError(f"{STEERING_CSV} not found.")

    df = pd.read_csv(STEERING_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Loaded {len(df):,} synoptic track points with 500hPa steering flow.")

    # Steering feature columns
    steering_cols = ["z_500", "u_500", "v_500", "u_700", "v_700"]

    # Impute any missing steering values
    for col in steering_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    X_steering_list = []
    Y_target_list = []
    metadata_list = []

    max_lead = max(lead_times)

    for cyclone_id, group in df.groupby("cyclone_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < history_len + max_lead:
            continue

        st_vals = group[steering_cols].values.astype(np.float32)
        coord_vals = group[["latitude", "longitude", "wind_speed_kmh"]].values.astype(np.float32)

        for i in range(len(group) - (history_len + max_lead) + 1):
            st_window = st_vals[i : i + history_len]
            targets = [coord_vals[i + history_len - 1 + dt] for dt in lead_times]

            X_steering_list.append(st_window)
            Y_target_list.append(targets)
            metadata_list.append({
                "cyclone_id": cyclone_id,
                "name": group.iloc[i + history_len - 1]["name"],
                "timestamp_t0": str(group.iloc[i + history_len - 1]["timestamp"]),
                "lat_t0": float(coord_vals[i + history_len - 1, 0]),
                "lon_t0": float(coord_vals[i + history_len - 1, 1]),
                "wind_t0": float(coord_vals[i + history_len - 1, 2])
            })

    X_steering = np.array(X_steering_list, dtype=np.float32)
    Y_target = np.array(Y_target_list, dtype=np.float32)
    meta_df = pd.DataFrame(metadata_list)

    print(f"Compiled Steering Sequences: {X_steering.shape} (N, 5, 5)")
    print(f"Compiled Target Sequences:   {Y_target.shape} (N, 3, 3)")

    out_npz = os.path.join(OUT_DIR, "steering_forecasting_sequences.npz")
    np.savez_compressed(out_npz, X_steering=X_steering, Y=Y_target)
    meta_df.to_csv(os.path.join(OUT_DIR, "sequences_metadata.csv"), index=False)

    print(f"Saved: {out_npz}")
    print(f"Saved: {os.path.join(OUT_DIR, 'sequences_metadata.csv')}")
    print("=" * 70)

if __name__ == "__main__":
    build_multimodal_dataset()
