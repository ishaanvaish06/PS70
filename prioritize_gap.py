"""
Prioritize the MOSDAC gap list: named storms first (more data, more
significant, usually better-documented), unnamed short-lived depressions
last (many are weak, brief systems with sparse wind/category data anyway).

Usage:
    python prioritize_gap.py
"""
import pandas as pd

df = pd.read_csv("data/metadata/mosdac_needed_cyclones.csv")
df["start"] = pd.to_datetime(df["start"])
df["end"] = pd.to_datetime(df["end"])
df["duration_days"] = (df["end"] - df["start"]).dt.total_seconds() / 86400

named = df[df["name"] != "UNNAMED"].sort_values("start")
unnamed = df[df["name"] == "UNNAMED"].sort_values("duration_days", ascending=False)

print(f"PRIORITY 1 - Named storms ({len(named)}): order these first")
print(named[["name", "season", "start", "end", "duration_days"]].to_string())

print(f"\nPRIORITY 2 - Unnamed systems ({len(unnamed)}): order only if time permits, longest-duration first")
print(unnamed[["name", "season", "start", "end", "duration_days"]].to_string())

named.to_csv("data/metadata/mosdac_priority1_named.csv", index=False)
unnamed.to_csv("data/metadata/mosdac_priority2_unnamed.csv", index=False)
print("\nSaved: mosdac_priority1_named.csv, mosdac_priority2_unnamed.csv")
