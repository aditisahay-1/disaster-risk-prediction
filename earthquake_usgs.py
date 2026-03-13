import pandas as pd
import glob

# path to all earthquake files
files = glob.glob("data/earthquake_*.csv")

print("Files found:", files)

# read all csv files
dfs = [pd.read_csv(file) for file in files]

# merge datasets
earthquakes = pd.concat(dfs, ignore_index=True)

print("Combined dataset shape:", earthquakes.shape)

# save merged dataset
earthquakes.to_csv("data/earthquakes.csv", index=False)

print("Merged dataset saved as earthquakes.csv")