import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# load dataset
df = pd.read_csv("data/disaster_ml_dataset.csv")

# features (reduced to avoid leakage)
X = df[
    [
        "earthquake_count",
        "avg_depth",
        "fire_count",
        "avg_frp"
    ]
]

# target
y = df["risk_level"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# predictions
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:")
print(classification_report(y_test, pred))

# save model for dashboard
joblib.dump(model, "risk_model.pkl")

print("\nModel saved as risk_model.pkl")