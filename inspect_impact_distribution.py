import pandas as pd

df = pd.read_csv("data/disaster_regression_dataset.csv")

print(df["impact_score"].describe())