"""
download_era5_pressure_levels.py
Downloads ERA5 Upper-Air Pressure Levels (500 hPa & 700 hPa)
from the Copernicus Climate Data Store (CDS API) for North Indian Ocean cyclone events.
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

VARIABLES = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
]

HOURS = ["00:00", "06:00", "12:00", "18:00"]

def build_monthly_requests(df, min_year=2018, max_year=2025):
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

def download_pressure_levels(dry_run=False, max_months=None, levels=["500", "700"], min_year=2018, max_year=2025):
    if not os.path.exists(IBTRACS_PATH):
        raise FileNotFoundError(f"{IBTRACS_PATH} not found.")

    df = pd.read_csv(IBTRACS_PATH)
    monthly_requests = build_monthly_requests(df, min_year=min_year, max_year=max_year)
    print(f"Total active cyclone months ({min_year}-{max_year}): {len(monthly_requests)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    client = None
    if not dry_run:
        if cdsapi is None:
            raise ImportError("cdsapi not installed. Run: pip install cdsapi")
        client = cdsapi.Client()

    count = 0
    # Process newest first (2024, 2023...)
    for (year, month), days in sorted(monthly_requests.items(), reverse=True):
        if max_months and count >= max_months:
            print(f"Reached limit of {max_months} months.")
            break

        out_path = os.path.join(OUT_DIR, f"era5_pl_{year}_{month:02d}.nc")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
            print(f"[skip] {out_path} already exists")
            continue

        request = {
            "product_type": "reanalysis",
            "variable": VARIABLES,
            "pressure_level": levels,
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": HOURS,
            "area": AREA,
            "data_format": "netcdf",
        }

        print(f"\nRequesting ERA5 500hPa/700hPa steering fields for {year}-{month:02d} ({len(days)} active days)...")
        if dry_run:
            print(f"  [DRY-RUN] Output: {out_path}")
            print(f"  Levels: {levels}, Variables: {VARIABLES}")
        else:
            client.retrieve("reanalysis-era5-pressure-levels", request, out_path)
            print(f"  Saved: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")
        count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print request plan without downloading")
    parser.add_argument("--max-months", type=int, default=None, help="Limit number of months")
    parser.add_argument("--min-year", type=int, default=2018, help="Starting year")
    parser.add_argument("--max-year", type=int, default=2025, help="Ending year")
    parser.add_argument("--levels", nargs="+", default=["500", "700"], help="Pressure levels (default: 500 700)")
    args = parser.parse_args()

    download_pressure_levels(
        dry_run=args.dry_run,
        max_months=args.max_months,
        levels=args.levels,
        min_year=args.min_year,
        max_year=args.max_year
    )
