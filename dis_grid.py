import pandas as pd

df = pd.read_csv("data/disaster_regression_dataset.csv")

print(df.groupby(["lat_grid","lon_grid"]).size().describe())