import pandas as pd
import numpy as np

df = pd.read_excel("data/emdat.xlsx")

# -----------------------------
# Filter disasters
# -----------------------------
selected_disasters = [
    "Earthquake",
    "Wildfire",
    "Flood",
    "Storm",
    "Drought",
    "Extreme temperature",
    "Landslide",
    "Volcanic activity"
]

df = df[df["Disaster Type"].isin(selected_disasters)]

# -----------------------------
# Select columns
# -----------------------------
df = df[
    [
        "Disaster Type",
        "Latitude",
        "Longitude",
        "Start Year",
        "Start Month",
        "Total Deaths",
        "Total Affected",
        "Total Damage ('000 US$)"
    ]
]

# -----------------------------
# Rename columns EARLY
# -----------------------------
df = df.rename(columns={
    "Start Year": "year",
    "Start Month": "month"
})

# -----------------------------
# Drop missing coordinates
# -----------------------------
df = df.dropna(subset=["Latitude", "Longitude"])

# -----------------------------
# Fill missing values
# -----------------------------
df["Total Deaths"] = df["Total Deaths"].fillna(0)
df["Total Affected"] = df["Total Affected"].fillna(0)
df["Total Damage ('000 US$)"] = df["Total Damage ('000 US$)"].fillna(0)

# -----------------------------
# Create impact score
# -----------------------------
df["impact_score"] = (
    np.log1p(df["Total Deaths"])
    + np.log1p(df["Total Affected"] / 1000)
    + np.log1p(df["Total Damage ('000 US$)"] / 1000)
)

# -----------------------------
# Create grid (VERY IMPORTANT)
# -----------------------------
df["lat_grid"] = (df["Latitude"] // 5) * 5
df["lon_grid"] = (df["Longitude"] // 5) * 5

# -----------------------------
# CREATE MULTI-DISASTER DATASET
# -----------------------------
pivot = df.pivot_table(
    index=["lat_grid", "lon_grid", "year", "month"],
    columns="Disaster Type",
    values="impact_score",
    aggfunc="mean"
).fillna(0)

all_disasters = [
    "Earthquake",
    "Wildfire",
    "Flood",
    "Storm",
    "Drought",
    "Extreme temperature",
    "Landslide",
    "Volcanic activity"
]

for disaster in all_disasters:
    if disaster not in pivot.columns:
        pivot[disaster] = 0

pivot = pivot.reset_index()

pivot.to_csv("data/emdat_multi_disaster_impact.csv", index=False)

print("✅ Multi-disaster dataset created")
print(pivot.head())
print("Shape:", pivot.shape)