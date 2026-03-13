import pandas as pd
import glob

files = glob.glob("data/firms_*.csv")

print("Files found:", files)

output_file = "data/wildfires.csv"

first = True

for file in files:
    print("Processing:", file)

    for chunk in pd.read_csv(file, chunksize=50000):
        if first:
            chunk.to_csv(output_file, index=False, mode="w")
            first = False
        else:
            chunk.to_csv(output_file, index=False, mode="a", header=False)

print("Merged wildfire dataset saved as wildfires.csv")