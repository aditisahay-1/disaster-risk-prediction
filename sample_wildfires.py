import pandas as pd

input_file = "data/wildfires.csv"
output_file = "data/wildfires_sampled.csv"

chunksize = 500000
sample_rate = 0.02   # keep 2% of rows

first = True

for chunk in pd.read_csv(input_file, chunksize=chunksize):
    
    sample = chunk.sample(frac=sample_rate)

    if first:
        sample.to_csv(output_file, index=False, mode="w")
        first = False
    else:
        sample.to_csv(output_file, index=False, mode="a", header=False)

print("Sampled wildfire dataset saved.")