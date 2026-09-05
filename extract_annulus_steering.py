"""
extract_annulus_steering.py
Extracts non-divergent environmental atmospheric steering flow from ERA5 500 hPa & 700 hPa
using an annular spatial ring (3 to 6 degree radius, ~330 to 660 km around cyclone center).
This eliminates inner vortex contamination and isolates the true background environmental steering flow.
"""

import os
import glob
import pandas as pd
import numpy as np
import xarray as xr

IBTRACS_PATH = os.path.join("data", "metadata", "ibtracs_clean.csv")
PL_DIR = os.path.join("data", "raw", "era5_pressure_levels")
OUT_CSV = os.path.join("data", "metadata", "ibtracs_with_steering.csv")
MULTIMODAL_DIR = os.path.join("data", "processed", "forecasting", "multimodal_sequences")
OUT_STEERING_NPZ = os.path.join(MULTIMODAL_DIR, "steering_forecasting_sequences.npz")
OUT_DATASET_NPZ = os.path.join(MULTIMODAL_DIR, "multimodal_forecasting_dataset.npz")

def compute_annulus_mean(sub_da, lat0, lon0, r_inner=3.0, r_outer=6.0):
    """Computes cosine-weighted annular average between r_inner and r_outer degrees."""
    lats = sub_da["latitude"].values
    lons = sub_da["longitude"].values
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    
    cos_lat = np.cos(np.radians(lat0))
    dist = np.sqrt((LAT - lat0)**2 + ((LON - lon0) * cos_lat)**2)
    mask = (dist >= r_inner) & (dist <= r_outer)
    
    if mask.sum() == 0:
        # Fallback to nearest if box too small or on boundary
        return float(sub_da.sel(latitude=lat0, longitude=lon0, method="nearest").values)
        
    vals = sub_da.values[mask]
    return float(np.nanmean(vals))

def extract_all_annulus_steering():
    print("=" * 70)
    print("ATMOSPHERIC ENVIRONMENTAL STEERING: 3°-6° ANNULUS EXTRACTION")
    print("=" * 70)

    files = sorted(glob.glob(os.path.join(PL_DIR, "era5_pl_*.nc")))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {PL_DIR}")
    print(f"Detected {len(files)} ERA5 pressure level files.")

    df = pd.read_csv(IBTRACS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    extracted_rows = []

    for f in files:
        base = os.path.basename(f).replace("era5_pl_", "").replace(".nc", "")
        parts = base.split("_")
        yr, mo = int(parts[0]), int(parts[1])

        month_pts = df[(df["year"] == yr) & (df["month"] == mo)]
        if len(month_pts) == 0:
            continue

        try:
            ds = xr.open_dataset(f)
            if "valid_time" in ds.coords and "time" not in ds.coords:
                ds = ds.rename({"valid_time": "time"})
            if "number" in ds.dims:
                ds = ds.isel(number=0, drop=True)
            if "expver" in ds.dims:
                ds = ds.isel(expver=0, drop=True)

            levels = [int(p) for p in ds["pressure_level"].values]
            has_500 = 500 in levels
            has_700 = 700 in levels

            ds_500 = ds.sel(pressure_level=500) if has_500 else None
            ds_700 = ds.sel(pressure_level=700) if has_700 else None

            for idx, row in month_pts.iterrows():
                t = row["timestamp"]
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                vals = row.to_dict()

                # Slice +/- 7 degrees box around cyclone
                lat_min, lat_max = min(lat - 7.5, lat + 7.5), max(lat - 7.5, lat + 7.5)
                lon_min, lon_max = lon - 7.5, lon + 7.5

                if ds_500 is not None:
                    t_500 = ds_500.sel(time=t, method="nearest")
                    sub_500 = t_500.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

                    vals["z_500"] = compute_annulus_mean(sub_500["z"], lat, lon) / 9.80665 # gpm
                    vals["u_500"] = compute_annulus_mean(sub_500["u"], lat, lon)          # m/s
                    vals["v_500"] = compute_annulus_mean(sub_500["v"], lat, lon)          # m/s
                else:
                    vals["z_500"], vals["u_500"], vals["v_500"] = np.nan, np.nan, np.nan

                if ds_700 is not None:
                    t_700 = ds_700.sel(time=t, method="nearest")
                    sub_700 = t_700.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

                    vals["u_700"] = compute_annulus_mean(sub_700["u"], lat, lon)          # m/s
                    vals["v_700"] = compute_annulus_mean(sub_700["v"], lat, lon)          # m/s
                else:
                    vals["u_700"], vals["v_700"] = np.nan, np.nan

                extracted_rows.append(vals)

            ds.close()
            print(f"  Processed {yr}-{mo:02d}: {len(month_pts)} track points extracted.")
        except Exception as e:
            print(f"  Error processing {f}: {e}")

    out_df = pd.DataFrame(extracted_rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(out_df):,} track points with non-divergent annulus steering to: {OUT_CSV}")

    # Now rebuild steering sequences
    print("\nRebuilding 5-step steering forecasting sequences...")
    build_sequences(out_df)

def build_sequences(df):
    os.makedirs(MULTIMODAL_DIR, exist_ok=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feature_cols = ["z_500", "u_500", "v_500", "u_700", "v_700"]

    sequences_X = []
    sequences_Y = []
    meta_rows = []

    for c_id, storm_df in df.groupby("cyclone_id"):
        storm_df = storm_df.sort_values("timestamp").reset_index(drop=True)
        if len(storm_df) < 13:
            continue

        for i in range(4, len(storm_df) - 8):
            # Input window: t-12h, t-9h, t-6h, t-3h, t (5 steps)
            x_win = storm_df.iloc[i-4 : i+1][feature_cols].values
            if np.isnan(x_win).any():
                continue

            # Targets at +6h (+2 steps), +12h (+4 steps), +24h (+8 steps)
            p6 = storm_df.iloc[i + 2]
            p12 = storm_df.iloc[i + 4]
            p24 = storm_df.iloc[i + 8]

            y_arr = np.array([
                [p6["latitude"], p6["longitude"], p6["wind_speed_kmh"]],
                [p12["latitude"], p12["longitude"], p12["wind_speed_kmh"]],
                [p24["latitude"], p24["longitude"], p24["wind_speed_kmh"]]
            ], dtype=np.float32)

            t0_row = storm_df.iloc[i]
            sequences_X.append(x_win.astype(np.float32))
            sequences_Y.append(y_arr)
            meta_rows.append({
                "cyclone_id": c_id,
                "name": t0_row["name"],
                "timestamp_t0": str(t0_row["timestamp"]),
                "lat_t0": t0_row["latitude"],
                "lon_t0": t0_row["longitude"],
                "wind_t0": t0_row["wind_speed_kmh"]
            })

    X_steering = np.array(sequences_X, dtype=np.float32)
    Y = np.array(sequences_Y, dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)

    print(f"Generated {len(X_steering)} valid steering sequences.")
    np.savez_compressed(OUT_STEERING_NPZ, X_steering=X_steering, Y=Y)
    meta_df.to_csv(os.path.join(MULTIMODAL_DIR, "sequences_metadata.csv"), index=False)
    print(f"Saved: {OUT_STEERING_NPZ}")

    # Update multimodal forecasting dataset with new annulus steering
    if os.path.exists(OUT_DATASET_NPZ):
        existing = np.load(OUT_DATASET_NPZ)
        np.savez_compressed(
            OUT_DATASET_NPZ,
            X_video=existing["X_video"][:len(X_steering)],
            X_steering=X_steering,
            curr_coords=existing["curr_coords"][:len(X_steering)],
            Y=Y,
            cyclone_ids=meta_df["cyclone_id"].values,
            names=meta_df["name"].values
        )
        print(f"Updated {OUT_DATASET_NPZ} with non-divergent annulus steering!")

if __name__ == "__main__":
    extract_all_annulus_steering()
