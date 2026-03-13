import pandas as pd

df = pd.read_csv("data/disaster_dataset_with_exposure.csv")

low_threshold = df["impact_score"].quantile(0.33)
high_threshold = df["impact_score"].quantile(0.66)

print("Low threshold:", low_threshold)
print("High threshold:", high_threshold)

def categorize(score):
    if score < low_threshold:
        return "Low"
    elif score < high_threshold:
        return "Medium"
    else:
        return "High"

df["impact_level"] = df["impact_score"].apply(categorize)

print(df["impact_level"].value_counts())

df.to_csv("data/disaster_dataset_with_exposure_levels.csv", index=False)

print("Dataset saved with exposure + impact levels")