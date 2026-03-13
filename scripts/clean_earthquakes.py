import pandas as pd

# load earthquakes
df = pd.read_csv("data/earthquakes.csv")

# keep useful columns
df = df[["time", "latitude", "longitude", "depth", "mag"]]

# convert time
df["time"] = pd.to_datetime(df["time"])

# extract temporal features
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month

# create spatial grid (5° bins)
df["lat_grid"] = (df["latitude"] // 5) * 5
df["lon_grid"] = (df["longitude"] // 5) * 5

df.to_csv("data/earthquakes_clean.csv", index=False)

print("Earthquake dataset cleaned.")
print(df.head())