"""
download_era5_pressure_levels.py
Downloads ERA5 Upper-Air Pressure Levels (500 hPa, 700 hPa, 850 hPa, 200 hPa)
from the Copernicus Climate Data Store (CDS API) for North Indian Ocean cyclone events.

Variables:
  - geopotential (z): Used to map the 500 hPa Subtropical Ridge position.
  - u_component_of_wind (u) & v_component_of_wind (v):
      * 500 hPa & 700 hPa: Deep-layer environmental steering currents.
      * 200 hPa - 850 hPa: 200-850 hPa Vertical Wind Shear calculation.

Usage:
  python download_era5_pressure_levels.py --dry-run
  python download_era5_pressure_levels.py
"""

import os
import sys
import argparse
import pandas as pd

try:
    import cdsapi
except ImportError:
    cdsapi = None

IBTRACS_PATH = os.path.join("data", "metadata", "ibtracs_clean.csv")
OUT_DIR = os.path.join("data", "raw", "era5_pressure_levels")

# North Indian Ocean bounding box [North, West, South, East]
AREA = [30, 50, -5, 105]

PRESSURE_LEVELS = ["200", "500", "700", "850"]

VARIABLES = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
]

HOURS = ["00:00", "06:00", "12:00", "18:00"]

def build_monthly_requests(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day

    requests = {}
    for (year, month), group in df.groupby(["year", "month"]):
        days = sorted(group["day"].unique())
        requests[(year, month)] = [f"{d:02d}" for d in days]
    return requests

def download_pressure_levels(dry_run=False, max_months=None):
    if not os.path.exists(IBTRACS_PATH):
        raise FileNotFoundError(f"{IBTRACS_PATH} not found.")

    df = pd.read_csv(IBTRACS_PATH)
    monthly_requests = build_monthly_requests(df)
    print(f"Total active cyclone months: {len(monthly_requests)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    client = None
    if not dry_run:
        if cdsapi is None:
            raise ImportError("cdsapi not installed. Run: pip install cdsapi")
        client = cdsapi.Client()

    count = 0
    for (year, month), days in sorted(monthly_requests.items(), reverse=True):
        if max_months and count >= max_months:
            print(f"Reached limit of {max_months} months.")
            break

        out_path = os.path.join(OUT_DIR, f"era5_pl_{year}_{month:02d}.nc")
        if os.path.exists(out_path):
            print(f"[skip] {out_path} already exists")
            continue

        request = {
            "product_type": "reanalysis",
            "variable": VARIABLES,
            "pressure_level": PRESSURE_LEVELS,
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": HOURS,
            "area": AREA,
            "data_format": "netcdf",
        }

        print(f"\nRequesting ERA5 pressure levels for {year}-{month:02d} ({len(days)} days)...")
        if dry_run:
            print(f"  [DRY-RUN] Output: {out_path}")
            print(f"  Levels: {PRESSURE_LEVELS}, Variables: {VARIABLES}")
        else:
            client.retrieve("reanalysis-era5-pressure-levels", request, out_path)
            print(f"  Saved: {out_path}")
        count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print request plan without downloading")
    parser.add_argument("--max-months", type=int, default=None, help="Limit number of months to download")
    args = parser.parse_args()

    download_pressure_levels(dry_run=args.dry_run, max_months=args.max_months)
