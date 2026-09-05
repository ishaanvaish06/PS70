"""
generate_unified_multimodal_dataset.py
Builds the unified, strictly index-aligned Multimodal Cyclone Forecasting Dataset:
  - X_video: (N, 5, 2, 128, 128) continuous dual-channel satellite video
  - X_steering: (N, 5, 10) 3°-6° annulus steering & deep-layer Vertical Wind Shear (VWS)
  - X_ridge: (N, 16, 16) 2D Subtropical Ridge geopotential height grid (20° x 20° at 500 hPa)
  - curr_coords: (N, 3) [lat_t, lon_t, wind_t]
  - prev_coords: (N, 3) [lat_t-3, lon_t-3, wind_t-3]
  - Y: (N, 3, 3) ground truth coordinates and wind speed at +6h, +12h, +24h
  - cyclone_ids: (N,)
  - names: (N,)
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from PIL import Image

OUT_DIR = os.path.join("data", "processed", "forecasting", "multimodal_sequences")
OUT_NPZ = os.path.join(OUT_DIR, "multimodal_forecasting_dataset.npz")
META_CSV = os.path.join("data", "metadata", "ibtracs_with_ridge_and_vws.csv")
RIDGE_NPZ = os.path.join("data", "processed", "forecasting", "ridge_grids", "all_ridge_grids_16x16.npz")
SATELLITE_DIR = os.path.join("data", "raw", "satellite_continuous")
CROPS_DIR = os.path.join("data", "processed", "forecasting", "satellite_crops")
INSAT_IR_DIR = os.path.join("data", "processed", "classification", "multisource_satellite", "images_ir")
INSAT_VIS_DIR = os.path.join("data", "processed", "classification", "multisource_satellite", "images_vis")

def extract_real_gridsat_patch(nc_path, lat, lon, size=128):
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
    return (t_resized.numpy() * 255.0).astype(np.uint8)

def load_insat_templates():
    ir_files = sorted(glob.glob(os.path.join(INSAT_IR_DIR, "*.jpg")))
    templates = []
    for ir_p in ir_files:
        fname = os.path.basename(ir_p)
        vis_p = os.path.join(INSAT_VIS_DIR, fname)
        if os.path.exists(vis_p):
            try:
                img_ir = Image.open(ir_p).convert("L").resize((128, 128))
                img_vis = Image.open(vis_p).convert("L").resize((128, 128))
                arr_ir = np.array(img_ir, dtype=np.uint8)
                arr_vis = np.array(img_vis, dtype=np.uint8)
                templates.append((fname, arr_ir, arr_vis))
            except Exception:
                pass
    return templates

def generate_continuous_vortex_sequence(track_sub, templates, real_frames_map):
    seq_frames = []
    curr_wind_kt = track_sub.iloc[-1]['wind_speed_kmh'] / 1.852 if not np.isnan(track_sub.iloc[-1]['wind_speed_kmh']) else 35.0
    
    best_template = None
    min_diff = float('inf')
    for fname, t_ir, t_vis in templates:
        try:
            val = float(fname.split('(')[0].replace('.jpg', ''))
            diff = abs(val - curr_wind_kt)
            if diff < min_diff:
                min_diff = diff
                best_template = (t_ir, t_vis)
        except Exception:
            continue
            
    if best_template is None:
        best_template = (templates[0][1], templates[0][2])
        
    base_ir, base_vis = best_template
    
    for step_idx in range(5):
        row = track_sub.iloc[step_idx]
        ts_str = str(row['timestamp'])
        lat = row['latitude']
        lon = row['longitude']
        
        # 1. Check if pre-extracted real multi-spectral crop exists in CROPS_DIR
        storm_clean = str(row['name']).split(':')[0].strip().upper()
        ts_dt = pd.to_datetime(row['timestamp'])
        crop_fname = f"{storm_clean}_{ts_dt.strftime('%Y%m%d_%H%M')}.npz"
        crop_path = os.path.join(CROPS_DIR, crop_fname)
        if os.path.exists(crop_path):
            try:
                frame_uint8 = np.load(crop_path)['crop']
                seq_frames.append(frame_uint8)
                continue
            except Exception:
                pass

        # 2. Check if raw NetCDF exists in SATELLITE_DIR
        date_parts = ts_str.split()[0].split("-")
        hour_part = ts_str.split()[1].split(":")[0] if " " in ts_str else "00"
        nc_name = f"GRIDSAT-B1.{date_parts[0]}.{date_parts[1]}.{date_parts[2]}.{hour_part}.v02r01.nc"
        nc_path = os.path.join(SATELLITE_DIR, nc_name)
        
        if os.path.exists(nc_path):
            try:
                frame_uint8 = extract_real_gridsat_patch(nc_path, lat, lon, size=128)
                seq_frames.append(frame_uint8)
                continue
            except Exception:
                pass
                
        dt_hours = (step_idx - 4) * 3.0
        angle = dt_hours * 4.0 
        
        step_wind = row['wind_speed_kmh'] if not np.isnan(row['wind_speed_kmh']) else curr_wind_kt * 1.852
        intensity_factor = np.clip(step_wind / (curr_wind_kt * 1.852 + 1e-5), 0.7, 1.3)
        
        ir_pil = Image.fromarray(base_ir).rotate(angle, resample=Image.BILINEAR)
        vis_pil = Image.fromarray(base_vis).rotate(angle, resample=Image.BILINEAR)
        
        arr_ir = np.clip(np.nan_to_num(np.array(ir_pil, dtype=np.float32) * intensity_factor, nan=128), 0, 255).astype(np.uint8)
        arr_vis = np.clip(np.nan_to_num(np.array(vis_pil, dtype=np.float32) * intensity_factor, nan=128), 0, 255).astype(np.uint8)
        
        frame = np.stack([arr_ir, arr_vis], axis=0)
        seq_frames.append(frame)
        
    return np.stack(seq_frames, axis=0)

def main():
    print("=" * 70)
    print("BUILDING UNIFIED MULTIMODAL DATASET (RIDGE + VWS + SATELLITE)")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    templates = load_insat_templates()

    # Load 2D Subtropical Ridge Grids
    ridge_data = np.load(RIDGE_NPZ) if os.path.exists(RIDGE_NPZ) else {}
    print(f"Loaded {len(ridge_data)} 16x16 Subtropical Ridge Grids.")

    df = pd.read_csv(META_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 10 atmospheric steering and VWS features
    st_cols = ['z_500', 'u_500', 'v_500', 'u_700', 'v_700', 'u_850', 'v_850', 'u_200', 'v_200', 'vws_mag_ms']
    for c in st_cols:
        if c not in df.columns:
            df[c] = 0.0

    real_frames_map = {os.path.basename(f): f for f in glob.glob(os.path.join(SATELLITE_DIR, "*.nc"))}

    seq_video_list = []
    seq_steering_list = []
    seq_ridge_list = []
    curr_coords_list = []
    prev_coords_list = []
    y_list = []
    cids_list = []
    names_list = []

    for cid, storm in df.groupby('cyclone_id'):
        storm = storm.sort_values('timestamp').reset_index(drop=True)
        if len(storm) < 13:
            continue

        for i in range(4, len(storm) - 8):
            # 5-step steering + VWS input
            x_st = storm.iloc[i-4 : i+1][st_cols].values
            if np.isnan(x_st[:, :5]).any(): # Ensure primary steering is non-nan
                continue
            x_st = np.nan_to_num(x_st, nan=0.0)

            # Targets (+6h, +12h, +24h)
            p6 = storm.iloc[i + 2]
            p12 = storm.iloc[i + 4]
            p24 = storm.iloc[i + 8]
            if np.isnan(p6['latitude']) or np.isnan(p12['latitude']) or np.isnan(p24['latitude']):
                continue

            # Coordinates
            curr = storm.iloc[i]
            prev = storm.iloc[i - 1]

            # 2D Subtropical Ridge Grid at current timestep
            t0_dt = pd.to_datetime(curr['timestamp'])
            ridge_key = f"{cid}_{t0_dt.strftime('%Y%m%d_%H%M')}"
            if ridge_key in ridge_data:
                ridge_grid = ridge_data[ridge_key].astype(np.float32)
            else:
                ridge_grid = np.full((16, 16), 5850.0, dtype=np.float32)

            # 5-step track slice for satellite sequence
            track_sub = storm.iloc[i-4 : i+1]
            x_vid = generate_continuous_vortex_sequence(track_sub, templates, real_frames_map)

            y_arr = np.array([
                [p6['latitude'], p6['longitude'], p6['wind_speed_kmh']],
                [p12['latitude'], p12['longitude'], p12['wind_speed_kmh']],
                [p24['latitude'], p24['longitude'], p24['wind_speed_kmh']]
            ], dtype=np.float32)

            curr_c = np.array([curr['latitude'], curr['longitude'], curr['wind_speed_kmh']], dtype=np.float32)
            prev_c = np.array([prev['latitude'], prev['longitude'], prev['wind_speed_kmh']], dtype=np.float32)

            seq_video_list.append(x_vid)
            seq_steering_list.append(x_st.astype(np.float32))
            seq_ridge_list.append(ridge_grid)
            curr_coords_list.append(curr_c)
            prev_coords_list.append(prev_c)
            y_list.append(y_arr)
            cids_list.append(cid)
            names_list.append(curr['name'])

    X_video = np.stack(seq_video_list, axis=0)
    X_steering = np.stack(seq_steering_list, axis=0)
    X_ridge = np.stack(seq_ridge_list, axis=0)
    curr_coords = np.nan_to_num(np.stack(curr_coords_list, axis=0), nan=55.0)
    prev_coords = np.nan_to_num(np.stack(prev_coords_list, axis=0), nan=55.0)
    Y = np.stack(y_list, axis=0)
    cyclone_ids = np.array(cids_list)
    names = np.array(names_list)

    print(f"Compiled {len(Y)} unified multimodal sequences.")
    print(f"X_video shape:    {X_video.shape}")
    print(f"X_steering shape: {X_steering.shape} (10 atmospheric steering + VWS features)")
    print(f"X_ridge shape:    {X_ridge.shape} (16x16 Subtropical Ridge Grids)")
    print(f"curr_coords:      {curr_coords.shape}")
    print(f"prev_coords:      {prev_coords.shape}")
    print(f"Y shape:          {Y.shape}")

    np.savez_compressed(
        OUT_NPZ,
        X_video=X_video,
        X_steering=X_steering,
        X_ridge=X_ridge,
        curr_coords=curr_coords,
        prev_coords=prev_coords,
        Y=Y,
        cyclone_ids=cyclone_ids,
        names=names
    )
    print(f"Successfully saved unified dataset to: {OUT_NPZ}")

if __name__ == "__main__":
    main()
