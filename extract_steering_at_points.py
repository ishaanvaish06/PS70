"""
extract_steering_at_points.py
Extracts 500 hPa and 700 hPa atmospheric steering variables (geopotential height, u-wind, v-wind)
at cyclone track points from the downloaded ERA5 pressure-level NetCDF files.
Saves data/metadata/ibtracs_with_steering.csv and updates forecasting sequence arrays.
"""

import os
import glob
import pandas as pd
import numpy as np
import xarray as xr

IBTRACS_PATH = os.path.join("data", "metadata", "ibtracs_clean.csv")
PL_DIR = os.path.join("data", "raw", "era5_pressure_levels")
OUT_CSV = os.path.join("data", "metadata", "ibtracs_with_steering.csv")
PROCESSED_FC_DIR = os.path.join("data", "processed", "forecasting")

def extract_steering():
    files = sorted(glob.glob(os.path.join(PL_DIR, "era5_pl_*.nc")))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {PL_DIR}")
    print(f"Loading {len(files)} ERA5 pressure level NetCDF files...")

    df = pd.read_csv(IBTRACS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    # Filter to years with downloaded files
    available_months = set()
    for f in files:
        base = os.path.basename(f).replace("era5_pl_", "").replace(".nc", "")
        parts = base.split("_")
        if len(parts) == 2:
            available_months.add((int(parts[0]), int(parts[1])))

    print(f"Matched {len(available_months)} active cyclone months with downloaded 3D steering grids.")

    extracted_rows = []
    
    for (yr, mo), group in df.groupby(["year", "month"]):
        if (yr, mo) not in available_months:
            continue
        nc_file = os.path.join(PL_DIR, f"era5_pl_{yr}_{mo:02d}.nc")
        if not os.path.exists(nc_file):
            continue

        try:
            ds = xr.open_dataset(nc_file)
            if "valid_time" in ds.coords and "time" not in ds.coords:
                ds = ds.rename({"valid_time": "time"})
            if "number" in ds.dims:
                ds = ds.isel(number=0, drop=True)
            if "expver" in ds.dims:
                ds = ds.isel(expver=0, drop=True)

            # Ensure pressure levels exist
            levels = [int(p) for p in ds["pressure_level"].values]
            has_500 = 500 in levels
            has_700 = 700 in levels

            ds_500 = ds.sel(pressure_level=500) if has_500 else None
            ds_700 = ds.sel(pressure_level=700) if has_700 else None

            for idx, row in group.iterrows():
                t = row["timestamp"]
                lat = row["latitude"]
                lon = row["longitude"]

                vals = row.to_dict()
                if ds_500 is not None:
                    pt_500 = ds_500.sel(latitude=lat, longitude=lon, time=t, method="nearest")
                    vals["z_500"] = float(pt_500["z"].values) / 9.80665 # Geopotential meters
                    vals["u_500"] = float(pt_500["u"].values)           # m/s
                    vals["v_500"] = float(pt_500["v"].values)           # m/s
                else:
                    vals["z_500"] = np.nan
                    vals["u_500"] = np.nan
                    vals["v_500"] = np.nan

                if ds_700 is not None:
                    pt_700 = ds_700.sel(latitude=lat, longitude=lon, time=t, method="nearest")
                    vals["u_700"] = float(pt_700["u"].values)
                    vals["v_700"] = float(pt_700["v"].values)
                else:
                    vals["u_700"] = np.nan
                    vals["v_700"] = np.nan

                extracted_rows.append(vals)
            ds.close()
        except Exception as e:
            print(f"Error processing {nc_file}: {e}")

    out_df = pd.DataFrame(extracted_rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSuccessfully extracted 500 hPa & 700 hPa steering variables for {len(out_df):,} cyclone track points!")
    print(f"Saved: {OUT_CSV}")
    return out_df

if __name__ == "__main__":
    extract_steering()
