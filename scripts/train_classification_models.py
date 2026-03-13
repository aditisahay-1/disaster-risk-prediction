import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv("data/disaster_filtered_classification_dataset.csv")

X = df[
    [
        "earthquake_count",
        "avg_magnitude",
        "avg_depth",
        "fire_count",
        "avg_brightness",
        "avg_frp"
    ]
]

y = df["impact_level"].map({"Low":0, "Medium":1, "High":2})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(f"\n{name}")
    print("Accuracy:", acc)

    joblib.dump(model, f"models/{name.replace(' ','_').lower()}_model.pkl")

# Select best model
best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]

print("\nBest Model:", best_model_name)
print("Best Accuracy:", best_accuracy)

best_model = models[best_model_name]

joblib.dump(best_model, "models/best_disaster_model.pkl")

print("Best model saved as best_disaster_model.pkl")