"""
Step 3 (part 2) — Person 1 (Data Engineer)
Extract ERA5 environmental values (SST, pressure, U/V wind) at each
cyclone observation's exact latitude/longitude/timestamp, from the
monthly NetCDF files downloaded by download_era5.py.

Usage:
    python extract_era5_at_points.py

Input:
    data/metadata/ibtracs_clean.csv
    data/raw/era5/era5_YYYY_MM.nc  (one or more)

Output:
    data/metadata/ibtracs_with_era5.csv
"""

import os
import glob
import pandas as pd
import xarray as xr

IBTRACS_PATH = os.path.join("data", "metadata", "ibtracs_clean.csv")
ERA5_DIR = os.path.join("data", "raw", "era5")
OUT_PATH = os.path.join("data", "metadata", "ibtracs_with_era5.csv")

# ERA5 variable names as they appear inside the NetCDF (short names differ
# from the request names above) -> the column name we want in our output.
VAR_RENAME = {
    "sst": "sst",
    "msl": "pressure_msl_pa",
    "u10": "u_wind",
    "v10": "v_wind",
}


def load_all_era5():
    files = sorted(glob.glob(os.path.join(ERA5_DIR, "era5_*.nc")))
    if not files:
        raise FileNotFoundError(
            f"No ERA5 files found in {ERA5_DIR} — run download_era5.py first.")
    print(f"Loading {len(files)} ERA5 file(s)...")
    # combine='by_coords' merges all months into one dataset along time
    ds = xr.open_mfdataset(files, combine="by_coords")
    ds = normalize_era5_dataset(ds)
    return ds


def normalize_era5_dataset(ds):
    """
    Since the 2024 CDS/Copernicus platform migration, downloaded NetCDF
    files use a different internal structure than the old API:
      - the time coordinate is named 'valid_time' instead of 'time'
      - there's often an extra 'number' dimension (ensemble member, size 1
        for reanalysis) and an 'expver' dimension (mixes final ERA5 with
        preliminary ERA5T data near the present).
    This function normalizes those away so the rest of the script can just
    use 'time', 'latitude', 'longitude' as before.
    """
    print(f"[debug] Raw ERA5 dims: {dict(ds.sizes)}")
    print(f"[debug] Raw ERA5 coords: {list(ds.coords)}")

    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
        print("[fix] Renamed 'valid_time' -> 'time'")

    if "number" in ds.dims:
        ds = ds.isel(number=0, drop=True)
        print("[fix] Dropped 'number' dimension (ensemble member 0)")

    if "expver" in ds.dims:
        # expver mixes final ERA5 (usually '0001') with preliminary ERA5T
        # ('0005') for the most recent ~5 days. Combine by taking whichever
        # expver has real (non-NaN) data at each point.
        if ds.sizes["expver"] > 1:
            combined = ds.isel(expver=0)
            for i in range(1, ds.sizes["expver"]):
                combined = combined.fillna(ds.isel(expver=i))
            ds = combined
            print(f"[fix] Combined {ds.sizes.get('expver', 1)} expver "
                  f"versions (final ERA5 + preliminary ERA5T) by fillna")
        else:
            ds = ds.isel(expver=0, drop=True)
            print("[fix] Dropped singleton 'expver' dimension")

    print(f"[debug] Normalized dims: {dict(ds.sizes)}\n")
    return ds


def extract_at_points(df, ds):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ERA5 longitude is 0-360 by convention in many CDS downloads; our
    # IBTrACS longitudes are -180..180. Normalize if needed.
    if float(ds["longitude"].max()) > 180:
        df["longitude_360"] = df["longitude"] % 360
    else:
        df["longitude_360"] = df["longitude"]

    results = {name: [] for name in VAR_RENAME.values()}
    n_missing = 0
    first_error = None

    for _, row in df.iterrows():
        try:
            point = ds.sel(
                time=row["timestamp"],
                latitude=row["latitude"],
                longitude=row["longitude_360"],
                method="nearest",
            )
            for era5_name, out_name in VAR_RENAME.items():
                if era5_name in point:
                    val = float(point[era5_name].values)
                    results[out_name].append(val)
                else:
                    results[out_name].append(None)
        except (KeyError, ValueError) as e:
            n_missing += 1
            if first_error is None:
                first_error = str(e)
            for out_name in VAR_RENAME.values():
                results[out_name].append(None)

    for out_name, values in results.items():
        df[out_name] = values

    if n_missing:
        print(f"[warn] First error seen ({n_missing} rows affected): {first_error}")

    # Convert units to the ones the rest of the team expects
    if "sst" in df.columns:
        df["sst_celsius"] = (df["sst"] - 273.15).round(2)  # Kelvin -> Celsius
    if "pressure_msl_pa" in df.columns:
        df["pressure_msl_hpa"] = (df["pressure_msl_pa"] / 100).round(1)  # Pa -> hPa

    df = df.drop(columns=["longitude_360", "sst", "pressure_msl_pa"], errors="ignore")

    print(f"Extracted ERA5 values for {len(df) - n_missing:,}/{len(df):,} rows "
          f"({n_missing} had no matching ERA5 grid point/time)")
    return df


def run():
    if not os.path.exists(IBTRACS_PATH):
        raise FileNotFoundError(f"{IBTRACS_PATH} not found — run get_ibtracs.py first.")

    df = pd.read_csv(IBTRACS_PATH)
    ds = load_all_era5()
    enriched = extract_at_points(df, ds)

    enriched.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH} ({len(enriched):,} rows)")
    print("\nSample:")
    print(enriched[["cyclone_id", "timestamp", "latitude", "longitude",
                     "sst_celsius", "pressure_msl_hpa", "u_wind", "v_wind"]].head())


if __name__ == "__main__":
    run()
