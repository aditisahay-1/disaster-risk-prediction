import numpy as np
import joblib
from tensorflow.keras.models import load_model

# load models
xgb_model = joblib.load("models/best_disaster_model.pkl")
lstm_model = load_model("models/lstm_model.keras")

# -----------------------------
# Hybrid prediction
# -----------------------------
def hybrid_predict(tabular_input, sequence_input):

    # XGBoost prediction
    xgb_pred = xgb_model.predict_proba([tabular_input])[0]

    # LSTM prediction
    lstm_pred = lstm_model.predict(sequence_input)[0]

    # combine (simple average)
    final_pred = (xgb_pred + lstm_pred) / 2

    return final_pred