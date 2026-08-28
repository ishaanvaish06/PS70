
import pandas as pd
df = pd.read_csv("data/metadata/ibtracs_clean.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["year"] = df["timestamp"].dt.year

print("Total rows:", len(df))
print("Rows with a category (wind data present):", df["category"].notna().sum())
print()
print("Rows per decade:")
print((df["year"] // 10 * 10).value_counts().sort_index())
print()
print("Rows from 2013 onward (INSAT-3D era):", (df["year"] >= 2013).sum())
print("Cyclones from 2013 onward:", df[df["year"] >= 2013]["cyclone_id"].nunique())
print("Of those, rows WITH category:", df[(df["year"] >= 2013) & df["category"].notna()].shape[0])
