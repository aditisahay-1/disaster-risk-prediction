import pandas as pd

df = pd.read_csv("data/wildfires_clean.csv")

agg = df.groupby(["lat_grid", "lon_grid", "year", "month"]).agg(
    fire_count=("brightness", "count"),
    avg_brightness=("brightness", "mean"),
    avg_frp=("frp", "mean")
).reset_index()

agg.to_csv("data/wildfire_features.csv", index=False)

print("Wildfire features created")
print(agg.head())