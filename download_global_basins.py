"""
download_global_basins.py
"""

import os
import sys
import pandas as pd
import numpy as np

RAW_DIR = os.path.join("data", "raw", "ibtracs_global")
PROCESSED_DIR = os.path.join("data", "processed", "forecasting")
COMBINED_CSV = os.path.join("data", "metadata", "ibtracs_global_wp_na.csv")

def build_forecasting_sequences(df, history_len=5, lead_times=[1, 2, 4]):
    X_list, Y_list = [], []
    max_lead = max(lead_times)
    
    for cyclone_id, group in df.groupby("cyclone_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        # Identify continuous segments where gap is <= 6.5 hours
        gap_mask = (group["timestamp"].diff().dt.total_seconds() > 6.5 * 3600.0)
        group["segment"] = gap_mask.cumsum()
        
        for _, seg in group.groupby("segment"):
            if len(seg) < history_len + max_lead:
                continue
            
            features = seg[["lat", "lon", "wind_speed"]].values.astype(np.float32)
            
            for i in range(len(seg) - (history_len + max_lead) + 1):
                x_window = features[i : i + history_len]
                y_targets = [features[i + history_len - 1 + dt] for dt in lead_times]
                X_list.append(x_window)
                Y_list.append(y_targets)
                
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y

def main():
    if not os.path.exists(COMBINED_CSV):
        print(f"Error: {COMBINED_CSV} not found.")
        return
        
    print(f"Loading {COMBINED_CSV}...")
    df = pd.read_csv(COMBINED_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Loaded {len(df):,} track observations from {df['cyclone_id'].nunique():,} cyclones.")
    
    print("Extracting global sliding-window sequences...")
    X_global, Y_global = build_forecasting_sequences(df)
    print(f"Generated Global Sequences: X = {X_global.shape}, Y = {Y_global.shape}")
    
    out_npz = os.path.join(PROCESSED_DIR, "global_pretrain_sequences.npz")
    np.savez_compressed(out_npz, X=X_global, Y=Y_global)
    print(f"Successfully saved global pre-training sequences to: {out_npz}")

if __name__ == "__main__":
    main()
