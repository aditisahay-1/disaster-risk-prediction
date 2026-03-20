import pandas as pd
import numpy as np
from sklearn.utils import resample

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/final_classification_dataset.csv")

# -----------------------------
# Add temporal features 🔥
# -----------------------------
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# -----------------------------
# Remove leakage columns
# -----------------------------
drop_cols = [
    "impact_level",
    "total_impact",
    "Earthquake", "Flood", "Storm", "Volcanic activity",
    "Wildfire", "Drought", "Extreme temperature", "Landslide"
]

# sort BEFORE feature extraction
df = df.sort_values(by=["lat_grid", "lon_grid", "year", "month"])

features = df.drop(columns=drop_cols)
print(features.columns)
# -----------------------------
# Create sequences 🔥
# -----------------------------
sequence_length = 6   # 🔥 increased

X_seq = []
y_seq = []

grouped = df.groupby(["lat_grid", "lon_grid"])

for _, group in grouped:
    group = group.reset_index(drop=True)

    if len(group) < sequence_length + 1:
        continue

    for i in range(len(group) - sequence_length):
        seq = group.iloc[i:i+sequence_length][features.columns].values
        label = group.iloc[i+sequence_length]["impact_level"]

        X_seq.append(seq)
        y_seq.append(label)

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq)

print("\nBefore balancing:")
print(pd.Series(y_seq).value_counts())

# -----------------------------
# 🔥 BALANCE DATA
# -----------------------------
X_flat = X_seq.reshape(X_seq.shape[0], -1)

df_seq = pd.DataFrame(X_flat)
df_seq["label"] = y_seq

df_low = df_seq[df_seq["label"] == "Low"]
df_medium = df_seq[df_seq["label"] == "Medium"]
df_high = df_seq[df_seq["label"] == "High"]

df_medium_up = resample(df_medium, replace=True, n_samples=5000, random_state=42)
df_high_up = resample(df_high, replace=True, n_samples=5000, random_state=42)

df_balanced = pd.concat([df_low, df_medium_up, df_high_up])
df_balanced = df_balanced.sample(frac=1, random_state=42)

print("\nAfter balancing:")
print(df_balanced["label"].value_counts())

# -----------------------------
# Convert back
# -----------------------------
label_map = {"Low": 0, "Medium": 1, "High": 2}

y_seq = df_balanced["label"].map(label_map).values.astype(np.int32)
X_seq = df_balanced.drop("label", axis=1).values.astype(np.float32)

num_features = X_seq.shape[1] // sequence_length
X_seq = X_seq.reshape(-1, sequence_length, num_features)

print("\nFinal shapes:")
print("X shape:", X_seq.shape)
print("y shape:", y_seq.shape)

# -----------------------------
# Save
# -----------------------------
np.save("data/X_lstm.npy", X_seq)
np.save("data/y_lstm.npy", y_seq)

print("\n✅ Balanced LSTM data saved")