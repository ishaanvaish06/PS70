"""
download_and_extract_satellite_archive.py
Downloads continuous 3-hourly geostationary satellite imagery (Thermal IR 11 um + Water Vapor 6.7 um)
for 32 major North Indian Ocean tropical cyclones (2013-2024) from NOAA GridSat-B1 (AWS Open Data archive).
Directly extracts storm-centered (2, 128, 128) multi-spectral patches and cleans up raw files,
scaling the library to 1,000+ real multi-spectral satellite sequence crops.
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

TARGET_STORMS = [
    # Super & Extremely Severe Cyclones (2013-2024)
    "PHAILIN", "HUDHUD", "VARDAH", "OCKHI", "TITLI", "GAJA", "FANI",
    "VAYU", "HIKAA", "KYARR", "MAHA", "BULBUL", "AMPHAN", "NISARGA",
    "GATI", "NIVAR", "BUREVI", "TAUKTAE", "YAAS", "GULAB", "SHAHEEN",
    "JAWAD", "ASANI", "SITRANG", "MANDOUS", "MOCHA", "BIPARJOY", "TEJ",
    "HAMOON", "MIDHILI", "MICHAUNG", "REMAL", "ASNA", "DANA"
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
        return False

def process_single_frame(task):
    storm_name, ts, lat, lon, nc_filename, url = task
    out_crop_file = os.path.join(CROPS_DIR, f"{storm_name}_{ts.strftime('%Y%m%d_%H%M')}.npz")
    
    if os.path.exists(out_crop_file) and os.path.getsize(out_crop_file) > 1000:
        return f"[EXISTS] {storm_name} at {ts}"

    tmp_nc_path = os.path.join(RAW_TMP_DIR, f"{storm_name}_{nc_filename}")
    try:
        t0 = time.time()
        urllib.request.urlretrieve(url, tmp_nc_path)
        dl_time = time.time() - t0

        ok = extract_and_save_crop(tmp_nc_path, lat, lon, out_crop_file)
        if os.path.exists(tmp_nc_path):
            os.remove(tmp_nc_path)

        if ok:
            return f"[SUCCESS] {storm_name} {ts} ({dl_time:.1f}s)"
        else:
            return f"[CROP FAILED] {storm_name} {ts}"
    except Exception as e:
        if os.path.exists(tmp_nc_path):
            try: os.remove(tmp_nc_path)
            except: pass
        return f"[ERROR] {storm_name} {ts}: {e}"

def build_task_list(max_frames_per_storm=32):
    df = pd.read_csv(IBTRACS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    tasks = []
    for storm_name in TARGET_STORMS:
        storm_df = df[df["name"].str.contains(storm_name, case=False, na=False)].sort_values("timestamp").reset_index(drop=True)
        if len(storm_df) == 0:
            continue

        # Sample up to max_frames_per_storm throughout storm lifecycle
        if len(storm_df) > max_frames_per_storm:
            step = len(storm_df) / float(max_frames_per_storm)
            indices = [int(round(i * step)) for i in range(max_frames_per_storm)]
            sub = storm_df.iloc[sorted(list(set(indices)))]
        else:
            sub = storm_df

        for _, row in sub.iterrows():
            ts = row["timestamp"]
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            yr, mo, dy, hr = ts.year, ts.month, ts.day, ts.hour

            nc_filename = f"GRIDSAT-B1.{yr}.{mo:02d}.{dy:02d}.{hr:02d}.v02r01.nc"
            url = f"{S3_BASE}/{yr}/{nc_filename}"
            tasks.append((storm_name, ts, lat, lon, nc_filename, url))

    return tasks

def download_batch(limit=None, workers=6):
    os.makedirs(RAW_TMP_DIR, exist_ok=True)
    os.makedirs(CROPS_DIR, exist_ok=True)

    tasks = build_task_list()
    print(f"Total candidate satellite frames: {len(tasks)} across {len(TARGET_STORMS)} major cyclones.")

    # Filter out already existing crops
    pending_tasks = []
    for t in tasks:
        sname, ts, _, _, _, _ = t
        out_crop = os.path.join(CROPS_DIR, f"{sname}_{ts.strftime('%Y%m%d_%H%M')}.npz")
        if not (os.path.exists(out_crop) and os.path.getsize(out_crop) > 1000):
            pending_tasks.append(t)

    print(f"Already downloaded on disk: {len(tasks) - len(pending_tasks)} frames.")
    print(f"Pending to download: {len(pending_tasks)} frames.")

    if limit and limit > 0:
        pending_tasks = pending_tasks[:limit]
        print(f"Downloading batch limit: {limit} frames.")

    if not pending_tasks:
        print("All requested frames are already cached.")
        return

    t_start = time.time()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_frame, task): task for task in pending_tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            completed += 1
            if completed % 10 == 0 or "SUCCESS" in res:
                print(f"  [{completed:03d}/{len(pending_tasks):03d}] {res}")

    total_time = time.time() - t_start
    print(f"Batch completed in {total_time:.1f}s. Total crops on disk: {len(os.listdir(CROPS_DIR))}")

if __name__ == "__main__":
    download_batch(limit=100, workers=6)
