"""
Step 1 — Person 1 (Data Engineer)
Download and clean IBTrACS track data for the North Indian Ocean basin
(the basin IMD is responsible for), producing a clean CSV that becomes
the historical ground-truth table for cyclone tracks + intensity.

Usage:
    python get_ibtracs.py

Output:
    data/raw/ibtracs/ibtracs_NI_raw.csv        <- untouched download
    data/metadata/ibtracs_clean.csv            <- cleaned, ready to join
                                                   with satellite/ERA5 data
"""

import os
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Official NOAA NCEI IBTrACS v04r01 endpoint, North Indian ("NI") basin subset.
# This basin already covers the storms IMD/RSMC New Delhi is responsible for,
# so we don't need to filter by country/basin ourselves.
IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
)

RAW_DIR = os.path.join("data", "raw", "ibtracs")
META_DIR = os.path.join("data", "metadata")
RAW_PATH = os.path.join(RAW_DIR, "ibtracs_NI_raw.csv")
CLEAN_PATH = os.path.join(META_DIR, "ibtracs_clean.csv")

# IBTrACS goes back to the 1840s, but INSAT-3D launched in 2013 (INSAT-3DR
# in 2016) — there is no real satellite imagery to pair with anything
# earlier. Confirmed via check_coverage.py: pre-2013 data is ~83% missing
# wind/category anyway, and can never be matched to imagery in Step 4.
# Keeping it would waste effort later, so we cut it here at the source.
YEAR_CUTOFF = 2013

# Columns we actually want. IBTrACS carries dozens of per-agency columns;
# we keep IMD's own (newdelhi_*) numbers where present since this is the
# basin IMD is authoritative for, and fall back to the WMO-designated
# agency figures (wmo_*) when IMD's own fields are blank.
WANTED_COLUMNS = [
    "SID",            # unique storm id, e.g. 2020329N10087
    "SEASON",         # year
    "NUMBER",         # storm number within season
    "BASIN",          # NI
    "SUBBASIN",       # BB (Bay of Bengal) / AS (Arabian Sea) etc.
    "NAME",           # storm name, or NOT_NAMED
    "ISO_TIME",       # UTC timestamp, 3-hourly for newdelhi obs
    "NATURE",         # DS/TS/ET/... storm nature
    "LAT",
    "LON",
    "WMO_WIND",       # wind per the currently-responsible WMO agency (kt)
    "WMO_PRES",       # pressure per the currently-responsible WMO agency (hPa)
    "WMO_AGENCY",
    "NEWDELHI_WIND",  # IMD's own 3-minute sustained wind (kt), when present
    "NEWDELHI_PRES",  # IMD's own central pressure (hPa), when present
    "NEWDELHI_CI",     # IMD's Dvorak current intensity, when present
]


def download_ibtracs():
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(RAW_PATH):
        print(f"[skip] Raw file already exists at {RAW_PATH}")
        return
    print(f"Downloading IBTrACS North Indian Ocean subset from:\n  {IBTRACS_URL}")
    try:
        df = pd.read_csv(IBTRACS_URL, low_memory=False)
    except Exception as e:
        print(
            "\n[ERROR] Could not download IBTrACS directly.\n"
            "If this is a network/firewall issue, download the file manually from:\n"
            f"  {IBTRACS_URL}\n"
            f"and save it to: {RAW_PATH}\n"
            "then re-run this script (it will pick up the local file and skip the download).\n"
        )
        raise e
    df.to_csv(RAW_PATH, index=False)
    print(f"Saved raw file: {RAW_PATH} ({len(df):,} rows)")


def clean_ibtracs():
    print("\nCleaning IBTrACS data...")
    # IBTrACS csv has a units row as row 0 (e.g. "YYYY","","","","","...", "deg","deg","kts","mb"...)
    # Read it once to detect that units row, then skip it.
    df = pd.read_csv(RAW_PATH, low_memory=False)
    if len(df) > 0 and str(df.iloc[0].get("SEASON", "")).strip().lower() in ("year", ""):
        # first row is metadata/units row, not real data
        df = df.iloc[1:].reset_index(drop=True)

    missing = [c for c in WANTED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[warn] Columns not found in this IBTrACS version, skipping: {missing}")
    keep_cols = [c for c in WANTED_COLUMNS if c in df.columns]
    df = df[keep_cols].copy()

    # Numeric coercion (IBTrACS stores these as strings, blanks for missing)
    for col in ["LAT", "LON", "WMO_WIND", "WMO_PRES", "NEWDELHI_WIND", "NEWDELHI_PRES"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")

    # Prefer IMD's own reported wind/pressure (this basin's responsible agency);
    # fall back to the generic WMO-designated figure when IMD's is missing.
    if "NEWDELHI_WIND" in df.columns:
        df["wind_speed_kt"] = df["NEWDELHI_WIND"].fillna(df.get("WMO_WIND"))
    else:
        df["wind_speed_kt"] = df.get("WMO_WIND")

    if "NEWDELHI_PRES" in df.columns:
        df["pressure_hpa"] = df["NEWDELHI_PRES"].fillna(df.get("WMO_PRES"))
    else:
        df["pressure_hpa"] = df.get("WMO_PRES")

    # Convert wind from knots -> km/h (IMD reports in km/h in bulletins/scale)
    df["wind_speed_kmh"] = (df["wind_speed_kt"] * 1.852).round(1)

    # Drop rows with no usable position or timestamp — can't be used downstream
    before = len(df)
    df = df.dropna(subset=["ISO_TIME", "LAT", "LON"]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with missing time/position "
          f"({before} -> {len(df)})")

    # Rename to the master-dataset schema used across the team
    df = df.rename(columns={
        "SID": "cyclone_id",
        "SEASON": "season",
        "NAME": "name",
        "SUBBASIN": "subbasin",
        "ISO_TIME": "timestamp",
        "LAT": "latitude",
        "LON": "longitude",
    })

    # IMD category is assigned from wind speed using the official
    # RSMC New Delhi scale (see classify_imd_category below) so every
    # row has a consistent category even where IBTrACS' own category
    # field is blank or uses a different agency's convention.
    df["category"] = df["wind_speed_kmh"].apply(classify_imd_category)

    final_cols = [
        "cyclone_id", "season", "name", "subbasin", "timestamp",
        "latitude", "longitude", "wind_speed_kmh", "pressure_hpa", "category",
    ]
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols]

    # Restrict to the INSAT-3D/3DR satellite era — see YEAR_CUTOFF note above.
    before_cutoff = len(df)
    before_cyclones = df["cyclone_id"].nunique()
    df = df[df["timestamp"].dt.year >= YEAR_CUTOFF].reset_index(drop=True)
    print(f"\nApplied satellite-era filter (>= {YEAR_CUTOFF}): "
          f"{before_cyclones} -> {df['cyclone_id'].nunique()} cyclones, "
          f"{before_cutoff:,} -> {len(df):,} rows")

    no_category = df["category"].isna().sum()
    if no_category:
        print(f"[note] {no_category} rows in this range still have no wind/category "
              f"(will need to rely on WMO_WIND or be excluded from classification training)")

    os.makedirs(META_DIR, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned file: {CLEAN_PATH} ({len(df):,} rows, "
          f"{df['cyclone_id'].nunique()} unique cyclones)")
    print("\nCategory distribution:")
    print(df["category"].value_counts())
    return df


def classify_imd_category(wind_kmh):
    """
    IMD / RSMC New Delhi official cyclone intensity scale, based on
    3-minute sustained surface wind speed in km/h.
    Reference: RSMC New Delhi tropical cyclone classification criteria.
    """
    if pd.isna(wind_kmh):
        return None
    if wind_kmh < 31:
        return "Low Pressure Area"
    elif wind_kmh < 50:
        return "Depression"
    elif wind_kmh < 62:
        return "Deep Depression"
    elif wind_kmh < 89:
        return "Cyclonic Storm"
    elif wind_kmh < 118:
        return "Severe Cyclonic Storm"
    elif wind_kmh < 166:
        return "Very Severe Cyclonic Storm"
    elif wind_kmh < 222:
        return "Extremely Severe Cyclonic Storm"
    else:
        return "Super Cyclonic Storm"


if __name__ == "__main__":
    download_ibtracs()
    clean_ibtracs()