import pandas as pd

# load hazard features
eq = pd.read_csv("data/earthquake_features.csv")
wf = pd.read_csv("data/wildfire_features.csv")

# load filtered EM-DAT impacts
impact = pd.read_csv("data/emdat_filtered_impact.csv")

# merge earthquake + wildfire features
hazards = pd.merge(
    eq,
    wf,
    on=["lat_grid", "lon_grid", "year", "month"],
    how="outer"
)

hazards = hazards.fillna(0)

# merge with impact dataset
df = pd.merge(
    hazards,
    impact,
    on=["lat_grid", "lon_grid", "year", "month"],
    how="inner"
)

df.to_csv("data/disaster_filtered_regression_dataset.csv", index=False)

print("Filtered regression dataset created")
print(df.head())
print("Shape:", df.shape)