import pandas as pd

# load final dataset
df = pd.read_csv("data/final_disaster_dataset.csv")

# -----------------------------
# Auto-detect impact columns
# -----------------------------
non_impact_cols = [
    "lat_grid", "lon_grid", "year", "month",
    "earthquake_count", "avg_magnitude", "avg_depth",
    "fire_count", "avg_brightness", "avg_frp",
    "prev_earthquake", "prev_fire"
]

impact_cols = [col for col in df.columns if col not in non_impact_cols]

print("Impact columns:", impact_cols)

# -----------------------------
# Total impact
# -----------------------------
df["total_impact"] = df[impact_cols].sum(axis=1)

# -----------------------------
# Create labels (balanced)
# -----------------------------
# -----------------------------
# Handle zero-heavy distribution
# -----------------------------

# -----------------------------
# Handle zero-heavy distribution
# -----------------------------

# initialize column as object (not categorical)
df["impact_level"] = None

# separate zero and non-zero
zero_mask = df["total_impact"] == 0
non_zero_mask = ~zero_mask

# assign LOW first
df.loc[zero_mask, "impact_level"] = "Low"

# create bins for non-zero values
df.loc[non_zero_mask, "impact_level"] = pd.qcut(
    df.loc[non_zero_mask, "total_impact"],
    q=2,
    labels=["Medium", "High"]
).astype(str)

# convert to category AFTER everything is assigned
df["impact_level"] = pd.Categorical(
    df["impact_level"],
    categories=["Low", "Medium", "High"]
)
# -----------------------------
# Save
# -----------------------------
df.to_csv("data/final_classification_dataset.csv", index=False)

print("\n✅ Classification dataset created")
print(df["impact_level"].value_counts())
print(df.head())