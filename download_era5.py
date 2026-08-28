"""
Step 3 — Person 1 (Data Engineer)
Download ERA5 environmental data (SST, mean sea-level pressure, U/V wind)
for the North Indian Ocean, restricted to only the dates our cyclones
actually occurred on. This avoids downloading the full global ERA5 archive.

Strategy: group cyclone observation dates by (year, month) and issue ONE
request per month covering only the specific days needed that month, over
a North Indian Ocean bounding box. This turns ~5,481 point-observations
into a handful of monthly requests instead of thousands of tiny ones.

Usage:
    python download_era5.py

Requires:
    pip install cdsapi
    A valid ~/.cdsapirc file (see registration steps)

Input:
    data/metadata/ibtracs_clean.csv

Output:
    data/raw/era5/era5_YYYY_MM.nc   (one NetCDF file per needed month)
"""

import os
import pandas as pd

try:
    import cdsapi
except ImportError:
    cdsapi = None  # allows --dry-run to work without the package installed

IBTRACS_PATH = os.path.join("data", "metadata", "ibtracs_clean.csv")
OUT_DIR = os.path.join("data", "raw", "era5")

# North Indian Ocean bounding box: [North, West, South, East]
# Covers Arabian Sea + Bay of Bengal with margin, not the whole globe.
AREA = [30, 50, -5, 105]

VARIABLES = [
    "sea_surface_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

# Cyclone obs are ~3-hourly; requesting these 8 synoptic hours covers them
# with room to nearest-match rather than requesting all 24 hourly slots.
HOURS = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]


def build_monthly_requests(df):
    """Return dict of {(year, month): sorted list of day strings}."""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day

    requests = {}
    for (year, month), group in df.groupby(["year", "month"]):
        days = sorted(group["day"].unique())
        requests[(year, month)] = [f"{d:02d}" for d in days]
    return requests


def download_era5(dry_run=False):
    if not os.path.exists(IBTRACS_PATH):
        raise FileNotFoundError(f"{IBTRACS_PATH} not found — run get_ibtracs.py first.")

    df = pd.read_csv(IBTRACS_PATH)
    monthly_requests = build_monthly_requests(df)

    print(f"Cyclone data spans {len(monthly_requests)} distinct (year, month) combinations.")
    print(f"This means {len(monthly_requests)} ERA5 API requests instead of "
          f"{len(df):,} individual point requests.\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    client = None
    if not dry_run:
        if cdsapi is None:
            raise ImportError("cdsapi not installed. Run: pip install cdsapi")
        client = cdsapi.Client()

    for (year, month), days in sorted(monthly_requests.items()):
        out_path = os.path.join(OUT_DIR, f"era5_{year}_{month:02d}.nc")
        if os.path.exists(out_path):
            print(f"[skip] {out_path} already exists")
            continue

        request = {
            "product_type": "reanalysis",
            "variable": VARIABLES,
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": HOURS,
            "area": AREA,
            "data_format": "netcdf",
        }

        print(f"Requesting {year}-{month:02d} "
              f"({len(days)} day(s): {', '.join(days)}) -> {out_path}")

        if dry_run:
            print(f"   [dry-run] request payload: {request}")
            continue

        try:
            client.retrieve("reanalysis-era5-single-levels", request, out_path)
            print(f"   Saved: {out_path}")
        except Exception as e:
            print(f"   [ERROR] Failed for {year}-{month:02d}: {e}")
            print("   If this is a licence error, go accept the licence on the "
                  "CDS website first (see registration steps).")

    print("\nDone. Each monthly file covers all needed days/hours for the North "
          "Indian Ocean box. Next: run extract_era5_at_points.py to pull the "
          "exact value at each cyclone's lat/lon/time.")


if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("Running in DRY-RUN mode: will print requests but not call the API.\n")
    download_era5(dry_run=dry_run)