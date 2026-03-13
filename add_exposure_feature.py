import pandas as pd

df = pd.read_csv("data/disaster_regression_dataset.csv")

# count disasters per grid cell
exposure = (
    df.groupby(["lat_grid", "lon_grid"])
    .size()
    .reset_index(name="historical_disaster_count")
)

# merge back into dataset
df = df.merge(exposure, on=["lat_grid", "lon_grid"], how="left")

df.to_csv("data/disaster_dataset_with_exposure.csv", index=False)

print("Exposure feature added")
print(df.head())
print("Shape:", df.shape)