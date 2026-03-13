import pandas as pd

# load datasets
eq = pd.read_csv("data/earthquake_features.csv")
wf = pd.read_csv("data/wildfire_features.csv")
impact = pd.read_csv("data/emdat_impact_features.csv")

# merge earthquake + wildfire features
hazards = pd.merge(
    eq,
    wf,
    on=["lat_grid", "lon_grid", "year", "month"],
    how="outer"
)

# fill missing hazard values
hazards = hazards.fillna(0)

# merge with impact dataset
df = pd.merge(
    hazards,
    impact,
    on=["lat_grid", "lon_grid", "year", "month"],
    how="inner"
)

df.to_csv("data/disaster_regression_dataset.csv", index=False)

print("Final regression dataset created")
print(df.head())
print("Shape:", df.shape)