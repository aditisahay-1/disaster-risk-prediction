import pandas as pd

df = pd.read_csv("data/disaster_features.csv")

# create risk score
df["risk_score"] = (
    df["earthquake_count"] * df["avg_magnitude"]
    + df["fire_count"] * (df["avg_brightness"] / 100)
)

# create risk levels
def categorize_risk(score):
    if score < 2:
        return "Low"
    elif score < 5:
        return "Medium"
    else:
        return "High"

df["risk_level"] = df["risk_score"].apply(categorize_risk)

df.to_csv("data/disaster_ml_dataset.csv", index=False)

print("ML dataset created")
print(df[["risk_score","risk_level"]].head())
print("Shape:", df.shape)