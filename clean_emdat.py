import pandas as pd
import numpy as np

df = pd.read_excel("data/emdat.xlsx")

# keep relevant columns
df = df[
    [
        "Latitude",
        "Longitude",
        "Start Year",
        "Start Month",
        "Total Deaths",
        "Total Affected",
        "Total Damage ('000 US$)"
    ]
]

# drop rows without coordinates
df = df.dropna(subset=["Latitude", "Longitude"])

# fill missing impact values
df["Total Deaths"] = df["Total Deaths"].fillna(0)
df["Total Affected"] = df["Total Affected"].fillna(0)
df["Total Damage ('000 US$)"] = df["Total Damage ('000 US$)"].fillna(0)

# compute impact score
df["impact_score"] = (
    np.log1p(df["Total Deaths"])
    + np.log1p(df["Total Affected"] / 1000)
    + np.log1p(df["Total Damage ('000 US$)"] / 1000)
)

# rename columns
df = df.rename(
    columns={
        "Start Year": "year",
        "Start Month": "month"
    }
)

# spatial grid
df["lat_grid"] = (df["Latitude"] // 5) * 5
df["lon_grid"] = (df["Longitude"] // 5) * 5

# aggregate by region/time
agg = df.groupby(
    ["lat_grid", "lon_grid", "year", "month"]
)["impact_score"].mean().reset_index()

# save
agg.to_csv("data/emdat_impact_features.csv", index=False)

print("EM-DAT impact dataset created")
print(agg.head())
print("Shape:", agg.shape)