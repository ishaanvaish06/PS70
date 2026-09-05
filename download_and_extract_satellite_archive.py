"""
download_and_extract_satellite_archive.py
Downloads continuous 3-hourly geostationary satellite imagery (Thermal IR 11 um + Water Vapor 6.7 um)
for 20 major North Indian Ocean tropical cyclones from NOAA GridSat-B1 (AWS Open Data archive).
Directly extracts storm-centered (2, 128, 128) multi-spectral patches and cleans up raw files.
"""

import os
import sys
import time
import urllib.request
import concurrent.futures
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

IBTRACS_PATH = os.path.join("data", "metadata", "ibtracs_clean.csv")
RAW_TMP_DIR = os.path.join("data", "raw", "satellite_continuous", "tmp_downloads")
CROPS_DIR = os.path.join("data", "processed", "forecasting", "satellite_crops")
S3_BASE = "https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/data"

# Target Major Named Cyclones (2018-2024)
TARGET_STORMS = [
    "AMPHAN", "MOCHA", "TAUKTAE", "YAAS", "REMAL",
    "BIPARJOY", "MICHAUNG", "DANA", "TEJ", "ASANI",
    "MANDOUS", "GULAB", "SHAHEEN", "ASNA", "NISARGA",
    "NIVAR", "BUREVI", "JAWAD", "FANI", "MAHA"
]

def extract_and_save_crop(nc_path, lat, lon, out_path, size=128):
    try:
        ds = xr.open_dataset(nc_path)
        ir = ds['irwin_cdr'].sel(lat=slice(lat - 4.5, lat + 4.5), lon=slice(lon - 4.5, lon + 4.5)).values[0]
        wv = ds['irwvp'].sel(lat=slice(lat - 4.5, lat + 4.5), lon=slice(lon - 4.5, lon + 4.5)).values[0]
        ds.close()

        ir = np.nan_to_num(ir, nan=295.0)
        wv = np.nan_to_num(wv, nan=245.0)

        ir_norm = np.clip((ir - 180.0) / 130.0, 0.0, 1.0)
        wv_norm = np.clip((wv - 200.0) / 70.0, 0.0, 1.0)

        t = torch.tensor(np.stack([ir_norm, wv_norm], axis=0), dtype=torch.float32).unsqueeze(0)
        t_resized = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
        crop_uint8 = (t_resized.numpy() * 255.0).astype(np.uint8)

        np.savez_compressed(out_path, crop=crop_uint8)
        return True
    except Exception as e:
        print(f"Error cropping {nc_path}: {e}")
        return False

def process_single_frame(task):
    """Downloads one frame, crops storm center, and removes raw file."""
    storm_name, ts, lat, lon, nc_filename, url = task
    out_crop_file = os.path.join(CROPS_DIR, f"{storm_name}_{ts.strftime('%Y%m%d_%H%M')}.npz")
    
    if os.path.exists(out_crop_file):
        return f"[EXISTS] {storm_name} at {ts}"

    tmp_nc_path = os.path.join(RAW_TMP_DIR, nc_filename)
    try:
        t0 = time.time()
        urllib.request.urlretrieve(url, tmp_nc_path)
        dl_time = time.time() - t0

        # Extract crop
        ok = extract_and_save_crop(tmp_nc_path, lat, lon, out_crop_file)
        
        # Cleanup temporary 35MB raw file
        if os.path.exists(tmp_nc_path):
            os.remove(tmp_nc_path)

        if ok:
            return f"[SUCCESS] {storm_name} {ts} (downloaded in {dl_time:.1f}s, cropped)"
        else:
            return f"[CROP FAILED] {storm_name} {ts}"
    except Exception as e:
        if os.path.exists(tmp_nc_path):
            try: os.remove(tmp_nc_path)
            except: pass
        return f"[ERROR] {storm_name} {ts}: {e}"

def build_task_list():
    df = pd.read_csv(IBTRACS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    tasks = []
    for storm_name in TARGET_STORMS:
        # Match storm name
        storm_df = df[df["name"].str.contains(storm_name, case=False, na=False)].sort_values("timestamp").reset_index(drop=True)
        if len(storm_df) == 0:
            continue

        # Find peak intensity index
        max_idx = storm_df["wind_speed_kmh"].idxmax()
        # Take 8 consecutive 3-hourly frames around peak intensity
        start_idx = max(0, max_idx - 4)
        end_idx = min(len(storm_df), start_idx + 8)
        sub = storm_df.iloc[start_idx:end_idx]

        for _, row in sub.iterrows():
            ts = row["timestamp"]
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            yr = ts.year
            mo = ts.month
            dy = ts.day
            hr = ts.hour

            nc_filename = f"GRIDSAT-B1.{yr}.{mo:02d}.{dy:02d}.{hr:02d}.v02r01.nc"
            url = f"{S3_BASE}/{yr}/{nc_filename}"
            tasks.append((storm_name, ts, lat, lon, nc_filename, url))

    return tasks

def main():
    print("=" * 70)
    print("CONTINUOUS SATELLITE ARCHIVE INGESTION (NOAA GRIDSAT-B1 / AWS OPEN DATA)")
    print("=" * 70)

    os.makedirs(RAW_TMP_DIR, exist_ok=True)
    os.makedirs(CROPS_DIR, exist_ok=True)

    tasks = build_task_list()
    print(f"Constructed task list: {len(tasks)} satellite frames across {len(TARGET_STORMS)} major cyclones.")
    print(f"Target Cyclones: {', '.join(TARGET_STORMS)}")
    print(f"Crops destination: {CROPS_DIR}\n")

    # Download and process concurrently with 4 workers
    max_workers = 4
    completed = 0
    t_start = time.time()

    print(f"Launching parallel extraction pool with {max_workers} threads...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_frame, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            completed += 1
            print(f"  [{completed:03d}/{len(tasks):03d}] {res}")

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"ALL FRAMES PROCESSED in {total_time/60:.2f} minutes!")
    print(f"Saved crops count: {len(os.listdir(CROPS_DIR))} files in {CROPS_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
