import pandas as pd

df = pd.read_csv("data/earthquakes_clean.csv")

agg = df.groupby(["lat_grid", "lon_grid", "year", "month"]).agg(
    earthquake_count=("mag", "count"),
    avg_magnitude=("mag", "mean"),
    avg_depth=("depth", "mean")
).reset_index()

agg.to_csv("data/earthquake_features.csv", index=False)

print("Earthquake features created")
print(agg.head())