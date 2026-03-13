import pandas as pd

# load datasets
eq = pd.read_csv("data/earthquake_features.csv")
wf = pd.read_csv("data/wildfire_features.csv")

# merge on spatial-temporal keys
df = pd.merge(
    eq,
    wf,
    on=["lat_grid", "lon_grid", "year", "month"],
    how="outer"
)

# fill missing values
df = df.fillna(0)

df.to_csv("data/disaster_features.csv", index=False)

print("Merged disaster feature dataset created")
print(df.head())
print("Shape:", df.shape)