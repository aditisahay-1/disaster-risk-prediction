import pandas as pd

df = pd.read_excel("data/emdat.xlsx")

filtered = df[df["Disaster Type"].isin(["Earthquake", "Wildfire"])]

print(filtered["Disaster Type"].value_counts())
print("Total rows:", len(filtered))