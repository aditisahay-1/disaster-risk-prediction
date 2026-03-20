import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from xgboost import XGBClassifier
import joblib

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/final_classification_dataset.csv")

# -----------------------------
# REMOVE DATA LEAKAGE 🚨
# -----------------------------
impact_cols = [
    "Earthquake",
    "Flood",
    "Storm",
    "Volcanic activity",
    "Wildfire",
    "Drought",
    "Extreme temperature",
    "Landslide"
]

X = df.drop(columns=[
    "impact_level",
    "lat_grid",
    "lon_grid",
    "year",
    "month",
    "total_impact"
] + impact_cols)

y = df["impact_level"].map({"Low": 0, "Medium": 1, "High": 2})

# -----------------------------
# 🔥 HANDLE IMBALANCE (UPSAMPLING)
# -----------------------------
df_combined = pd.concat([X, y.rename("impact_level")], axis=1)

df_low = df_combined[df_combined["impact_level"] == 0]
df_medium = df_combined[df_combined["impact_level"] == 1]
df_high = df_combined[df_combined["impact_level"] == 2]

print("Before balancing:")
print(df_combined["impact_level"].value_counts())

# Upsample minority classes
df_medium_up = resample(df_medium, replace=True, n_samples=5000, random_state=42)
df_high_up = resample(df_high, replace=True, n_samples=5000, random_state=42)

# Combine
df_balanced = pd.concat([df_low, df_medium_up, df_high_up])

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42)

print("\nAfter balancing:")
print(df_balanced["impact_level"].value_counts())

# Split back
X = df_balanced.drop("impact_level", axis=1)
y = df_balanced["impact_level"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Feature Scaling (for LR)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )
}

# -----------------------------
# Training & Evaluation
# -----------------------------
results = {}

for name, model in models.items():

    print(f"\n==================== {name} ====================")

    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print("Accuracy:", acc)
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # Save model
    filename = f"models/{name.replace(' ','_').lower()}_model.pkl"
    joblib.dump(model, filename)

# -----------------------------
# Best model selection
# -----------------------------
best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]

print("\n===================================")
print("Best Model:", best_model_name)
print("Best Accuracy:", best_accuracy)

best_model = models[best_model_name]
joblib.dump(best_model, "models/best_disaster_model.pkl")

print("✅ Best model saved as best_disaster_model.pkl")