import pandas as pd

df = pd.read_excel("data/emdat.xlsx")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nSample rows:")
print(df.head())

print("\nMissing values:")
print(df[["Latitude","Longitude","Total Deaths","Total Affected","Total Damage ('000 US$)"]].isna().sum())