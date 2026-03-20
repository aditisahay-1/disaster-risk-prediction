import pandas as pd

# -----------------------------
# Load hazard features
# -----------------------------
eq = pd.read_csv("data/earthquake_features.csv")
wf = pd.read_csv("data/wildfire_features.csv")

# -----------------------------
# Merge hazard features
# -----------------------------
hazards = pd.merge(
    eq,
    wf,
    on=["lat_grid", "lon_grid", "year", "month"],
    how="outer"
)

hazards = hazards.fillna(0)

# -----------------------------
# Load MULTI-DISASTER impact
# -----------------------------
impact = pd.read_csv("data/emdat_multi_disaster_impact.csv")

# -----------------------------
# Merge hazards + impact
# -----------------------------
df = pd.merge(
    hazards,
    impact,
    on=["lat_grid", "lon_grid", "year", "month"],
    how="left"   # 🔥 VERY IMPORTANT
)

# -----------------------------
# Fill missing values
# -----------------------------
df = df.fillna(0)

# -----------------------------
# Sort for time-series
# -----------------------------
df = df.sort_values(by=["lat_grid", "lon_grid", "year", "month"])

# -----------------------------
# Add temporal feature (lag)
# -----------------------------
df["prev_earthquake"] = df.groupby(["lat_grid","lon_grid"])["earthquake_count"].shift(1)
df["prev_fire"] = df.groupby(["lat_grid","lon_grid"])["fire_count"].shift(1)

df["prev_earthquake"] = df["prev_earthquake"].fillna(0)
df["prev_fire"] = df["prev_fire"].fillna(0)

# -----------------------------
# Save FINAL dataset
# -----------------------------
df.to_csv("data/final_disaster_dataset.csv", index=False)

print("✅ Final dataset created")
print(df.head())
print("Shape:", df.shape)