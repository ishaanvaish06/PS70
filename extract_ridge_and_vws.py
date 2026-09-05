"""
extract_ridge_and_vws.py
Extracts:
  1. 2D Subtropical Ridge Grid (20° x 20° spatial grid of 500 hPa geopotential height, resized to 16x16)
     directly from existing ERA5 pressure-level NetCDF files.
  2. Deep-layer Vertical Wind Shear (VWS: 200 hPa - 850 hPa wind vector, magnitude, and components)
     from downloaded ERA5 VWS NetCDF files.
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

IBTRACS_PATH = os.path.join("data", "metadata", "ibtracs_with_steering.csv")
PL_DIR = os.path.join("data", "raw", "era5_pressure_levels")
VWS_DIR = os.path.join("data", "raw", "era5_vws")
OUT_CSV = os.path.join("data", "metadata", "ibtracs_with_ridge_and_vws.csv")
RIDGE_NPZ_DIR = os.path.join("data", "processed", "forecasting", "ridge_grids")

def compute_annulus_mean(sub_da, lat0, lon0, r_inner=3.0, r_outer=6.0):
    """Computes annular average between r_inner and r_outer degrees."""
    lats = sub_da["latitude"].values
    lons = sub_da["longitude"].values
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    cos_lat = np.cos(np.radians(lat0))
    dist = np.sqrt((LAT - lat0)**2 + ((LON - lon0) * cos_lat)**2)
    mask = (dist >= r_inner) & (dist <= r_outer)
    if mask.sum() == 0:
        return float(sub_da.sel(latitude=lat0, longitude=lon0, method="nearest").values)
    vals = sub_da.values[mask]
    return float(np.nanmean(vals))

def extract_ridge_grid_16x16(ds_500, time_val, lat0, lon0, box_deg=10.0):
    """
    Extracts a 20° x 20° bounding box centered on (lat0, lon0)
    and interpolates to a fixed (16, 16) spatial grid of geopotential meters (gpm).
    """
    lat_min = max(-5.0, lat0 - box_deg)
    lat_max = min(30.0, lat0 + box_deg)
    lon_min = max(50.0, lon0 - box_deg)
    lon_max = min(105.0, lon0 + box_deg)

    t_da = ds_500["z"].sel(time=time_val, method="nearest")
    # ERA5 latitudes are ordered descending (30 to -5)
    patch = t_da.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)).values / 9.80665

    patch = np.nan_to_num(patch, nan=5850.0)

    # Convert to torch tensor and interpolate to 16x16
    t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t_16 = F.interpolate(t, size=(16, 16), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    return t_16.numpy().astype(np.float32)

def extract_all_ridge_and_vws():
    print("=" * 70)
    print("EXTRACTING 2D SUBTROPICAL RIDGE GRIDS (20°x20°) & VERTICAL WIND SHEAR (VWS)")
    print("=" * 70)

    os.makedirs(RIDGE_NPZ_DIR, exist_ok=True)
    df = pd.read_csv(IBTRACS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    pl_files = {os.path.basename(f): f for f in glob.glob(os.path.join(PL_DIR, "era5_pl_*.nc"))}
    vws_files = {os.path.basename(f): f for f in glob.glob(os.path.join(VWS_DIR, "era5_vws_*.nc")) if os.path.getsize(f) > 500000}

    print(f"Found {len(pl_files)} ERA5 500hPa steering files on disk.")
    print(f"Found {len(vws_files)} ERA5 200/850hPa VWS files on disk.")

    extracted_rows = []
    ridge_cache = {}

    for (yr, mo), group in df.groupby(["year", "month"]):
        pl_name = f"era5_pl_{yr}_{mo:02d}.nc"
        vws_name = f"era5_vws_{yr}_{mo:02d}.nc"

        if pl_name not in pl_files:
            continue

        try:
            ds_pl = xr.open_dataset(pl_files[pl_name])
            if "valid_time" in ds_pl.coords and "time" not in ds_pl.coords:
                ds_pl = ds_pl.rename({"valid_time": "time"})
            if "number" in ds_pl.dims:
                ds_pl = ds_pl.isel(number=0, drop=True)
            if "expver" in ds_pl.dims:
                ds_pl = ds_pl.isel(expver=0, drop=True)

            ds_500 = ds_pl.sel(pressure_level=500) if 500 in [int(p) for p in ds_pl["pressure_level"].values] else None

            # Open VWS file if available
            ds_vws = None
            if vws_name in vws_files:
                try:
                    ds_v = xr.open_dataset(vws_files[vws_name])
                    if "valid_time" in ds_v.coords and "time" not in ds_v.coords:
                        ds_v = ds_v.rename({"valid_time": "time"})
                    if "number" in ds_v.dims:
                        ds_v = ds_v.isel(number=0, drop=True)
                    if "expver" in ds_v.dims:
                        ds_v = ds_v.isel(expver=0, drop=True)
                    ds_vws = ds_v
                except Exception as e:
                    print(f"  Warning opening {vws_name}: {e}")

            for idx, row in group.iterrows():
                t = row["timestamp"]
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                c_id = row["cyclone_id"]
                vals = row.to_dict()

                # 1. 2D Subtropical Ridge Grid
                if ds_500 is not None:
                    ridge_grid = extract_ridge_grid_16x16(ds_500, t, lat, lon, box_deg=10.0)
                    key = f"{c_id}_{pd.to_datetime(t).strftime('%Y%m%d_%H%M')}"
                    ridge_cache[key] = ridge_grid

                    # Summary statistics for metadata CSV
                    vals["ridge_mean_gpm"] = float(np.mean(ridge_grid))
                    vals["ridge_max_gpm"] = float(np.max(ridge_grid))
                    vals["ridge_grad_y"] = float(np.mean(ridge_grid[0:4, :]) - np.mean(ridge_grid[12:16, :])) # North-South slope
                    vals["ridge_grad_x"] = float(np.mean(ridge_grid[:, 12:16]) - np.mean(ridge_grid[:, 0:4])) # East-West slope
                else:
                    vals["ridge_mean_gpm"] = 5850.0
                    vals["ridge_max_gpm"] = 5880.0
                    vals["ridge_grad_y"] = 0.0
                    vals["ridge_grad_x"] = 0.0

                # 2. Vertical Wind Shear (200 hPa vs 850 hPa)
                if ds_vws is not None:
                    try:
                        vws_levels = [int(p) for p in ds_vws["pressure_level"].values]
                        t_vws = ds_vws.sel(time=t, method="nearest")

                        lat_min, lat_max = min(lat - 7.5, lat + 7.5), max(lat - 7.5, lat + 7.5)
                        lon_min, lon_max = lon - 7.5, lon + 7.5

                        sub_200 = t_vws.sel(pressure_level=200, latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                        sub_850 = t_vws.sel(pressure_level=850, latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

                        u200 = compute_annulus_mean(sub_200["u"], lat, lon)
                        v200 = compute_annulus_mean(sub_200["v"], lat, lon)
                        u850 = compute_annulus_mean(sub_850["u"], lat, lon)
                        v850 = compute_annulus_mean(sub_850["v"], lat, lon)

                        u_shear = u200 - u850
                        v_shear = v200 - v850
                        vws_mag = np.sqrt(u_shear**2 + v_shear**2)

                        vals["u_200"] = u200
                        vals["v_200"] = v200
                        vals["u_850"] = u850
                        vals["v_850"] = v850
                        vals["u_shear"] = u_shear
                        vals["v_shear"] = v_shear
                        vals["vws_mag_ms"] = vws_mag
                    except Exception as e:
                        # Fallback
                        vals["u_200"] = 0.0
                        vals["v_200"] = 0.0
                        vals["u_850"] = 0.0
                        vals["v_850"] = 0.0
                        vals["u_shear"] = 0.0
                        vals["v_shear"] = 0.0
                        vals["vws_mag_ms"] = 0.0
                else:
                    # Climatological defaults if VWS file pending
                    vals["u_200"] = 0.0
                    vals["v_200"] = 0.0
                    vals["u_850"] = 0.0
                    vals["v_850"] = 0.0
                    vals["u_shear"] = 0.0
                    vals["v_shear"] = 0.0
                    vals["vws_mag_ms"] = 0.0

                extracted_rows.append(vals)

            ds_pl.close()
            if ds_vws is not None:
                ds_vws.close()

            print(f"  Processed {yr}-{mo:02d}: {len(group)} track points extracted.")
        except Exception as e:
            print(f"  Error processing {pl_name}: {e}")

    out_df = pd.DataFrame(extracted_rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved enriched track metadata to: {OUT_CSV}")

    # Save compact ridge grids
    ridge_npz_path = os.path.join(RIDGE_NPZ_DIR, "all_ridge_grids_16x16.npz")
    np.savez_compressed(ridge_npz_path, **ridge_cache)
    print(f"Saved {len(ridge_cache)} 16x16 Subtropical Ridge Grids to: {ridge_npz_path}")

if __name__ == "__main__":
    extract_all_ridge_and_vws()
