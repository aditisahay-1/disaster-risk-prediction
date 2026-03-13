import pandas as pd

df = pd.read_excel("data/emdat.xlsx")

print(df["Disaster Type"].value_counts())