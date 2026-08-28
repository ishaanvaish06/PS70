"""
Check how much of ibtracs_clean.csv the Kaggle 2013-2021 dataset can cover,
vs. how much needs to come from MOSDAC (2022 onward) or be accepted as a gap.

Usage:
    python split_coverage_gap.py
"""
import pandas as pd

df = pd.read_csv("data/metadata/ibtracs_clean.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["year"] = df["timestamp"].dt.year

KAGGLE_CUTOFF = 2021  # Kaggle dataset covers 2013-2021

covered = df[df["year"] <= KAGGLE_CUTOFF]
gap = df[df["year"] > KAGGLE_CUTOFF]

print(f"Cyclones coverable by Kaggle (2013-{KAGGLE_CUTOFF}): {covered['cyclone_id'].nunique()}")
print(f"Cyclones in the gap (2022 onward):                  {gap['cyclone_id'].nunique()}")
print()
print("Gap cyclones (name, season, date range) -- these need MOSDAC or another source:")
gap_summary = gap.groupby("cyclone_id").agg(
    name=("name", "first"),
    season=("season", "first"),
    start=("timestamp", "min"),
    end=("timestamp", "max"),
).sort_values("start")
print(gap_summary.to_string())

gap_summary.to_csv("data/metadata/mosdac_needed_cyclones.csv")
print("\nSaved list to: data/metadata/mosdac_needed_cyclones.csv")
print("Use the start/end dates per storm to order small, targeted MOSDAC")
print("requests instead of one huge date range.")
