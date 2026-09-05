"""
download_era5_vws.py
Downloads ERA5 Upper-Air Pressure Levels (200 hPa & 850 hPa)
from Copernicus Climate Data Store (CDS API) for computing Vertical Wind Shear (VWS).
"""

import os
import sys
import argparse
import pandas as pd
import cdsapi

IBTRACS_PATH = os.path.join("data", "metadata", "ibtracs_clean.csv")
OUT_DIR = os.path.join("data", "raw", "era5_vws")

AREA = [30, 50, -5, 105]

VARIABLES = [
    "u_component_of_wind",
    "v_component_of_wind",
]

LEVELS = ["200", "850"]
HOURS = ["00:00", "06:00", "12:00", "18:00"]

def get_monthly_requests(min_year=2020, max_year=2024):
    df = pd.read_csv(IBTRACS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day

    df = df[(df["year"] >= min_year) & (df["year"] <= max_year)]
    
    requests = {}
    for (year, month), group in df.groupby(["year", "month"]):
        days = sorted(group["day"].unique())
        requests[(year, month)] = [f"{d:02d}" for d in days]
    return requests

def download_all_vws(min_year=2020, max_year=2024, dry_run=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    monthly_reqs = get_monthly_requests(min_year, max_year)
    print(f"Total active cyclone months ({min_year}-{max_year}): {len(monthly_reqs)}")

    client = None if dry_run else cdsapi.Client()

    for (year, month), days in sorted(monthly_reqs.items(), reverse=True):
        out_path = os.path.join(OUT_DIR, f"era5_vws_{year}_{month:02d}.nc")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100000:
            print(f"[skip] {out_path} already exists ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")
            continue

        request = {
            "product_type": "reanalysis",
            "variable": VARIABLES,
            "pressure_level": LEVELS,
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": HOURS,
            "area": AREA,
            "data_format": "netcdf",
        }

        print(f"\nRequesting ERA5 200hPa/850hPa winds for {year}-{month:02d} ({len(days)} active days)...")
        if dry_run:
            print(f"  [DRY-RUN] Would save to: {out_path}")
        else:
            client.retrieve("reanalysis-era5-pressure-levels", request, out_path)
            print(f"  Saved: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-year", type=int, default=2020)
    parser.add_argument("--max-year", type=int, default=2024)
    args = parser.parse_args()
    download_all_vws(args.min_year, args.max_year, args.dry_run)
