import pandas as pd

df = pd.read_csv("data/wildfires_sampled.csv")

# keep useful columns
df = df[[
    "latitude",
    "longitude",
    "brightness",
    "confidence",
    "frp",
    "acq_date"
]]

# convert time
df["acq_date"] = pd.to_datetime(df["acq_date"])

df["year"] = df["acq_date"].dt.year
df["month"] = df["acq_date"].dt.month

# spatial grid
df["lat_grid"] = (df["latitude"] // 5) * 5
df["lon_grid"] = (df["longitude"] // 5) * 5

df.to_csv("data/wildfires_clean.csv", index=False)

print("Wildfire dataset cleaned.")
print(df.head())