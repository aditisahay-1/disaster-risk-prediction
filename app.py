import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Disaster Risk Prediction", layout="wide")

st.title("🌍 Multi-Hazard Disaster Risk Prediction System")
st.write("Hybrid Model: XGBoost + LSTM (Spatio-Temporal Intelligence)")

# -------------------------
# Load models
# -------------------------
xgb_model = joblib.load("models/best_disaster_model.pkl")
lstm_model = load_model("models/lstm_model.keras")

# -------------------------
# Load dataset
# -------------------------
data = pd.read_csv("data/final_classification_dataset.csv")

# Add temporal features
data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

# -------------------------
# Geocoder
# -------------------------
geolocator = Nominatim(user_agent="disaster_app")

# -------------------------
# Session state
# -------------------------
if "risk" not in st.session_state:
    st.session_state.risk = None

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("📍 Location Selection")

city_query = st.sidebar.text_input("Enter Location")

locations = []

if city_query:
    try:
        results = geolocator.geocode(city_query, exactly_one=False, limit=5)
        if results:
            for loc in results:
                locations.append((loc.address, loc.latitude, loc.longitude))
    except:
        pass

selected_location = None

if locations:
    selected_location = st.sidebar.selectbox(
        "Select Location",
        options=locations,
        format_func=lambda x: x[0]
    )

predict_button = st.sidebar.button("🚨 Predict Risk")

# -------------------------
# Helpers
# -------------------------
def get_grid(lat, lon):
    return round(lat / 5) * 5, round(lon / 5) * 5


def get_tabular_features(lat_grid, lon_grid):
    subset = data[
        (data["lat_grid"] == lat_grid) &
        (data["lon_grid"] == lon_grid)
    ]

    if subset.empty:
        return None

    row = subset.iloc[-1]

    return [
        row["earthquake_count"],
        row["avg_magnitude"],
        row["avg_depth"],
        row["fire_count"],
        row["avg_brightness"],
        row["avg_frp"],
        row["prev_earthquake"],
        row["prev_fire"]
    ]


def get_sequence(lat_grid, lon_grid, seq_len=6):
    subset = data[
        (data["lat_grid"] == lat_grid) &
        (data["lon_grid"] == lon_grid)
    ].sort_values(by=["year", "month"])

    if len(subset) < seq_len:
        return None

    feature_cols = [
        "lat_grid", "lon_grid", "year", "month",
        "earthquake_count", "avg_magnitude", "avg_depth",
        "fire_count", "avg_brightness", "avg_frp",
        "prev_earthquake", "prev_fire",
        "month_sin", "month_cos"
    ]

    for col in feature_cols:
        if col not in subset.columns:
            subset[col] = 0

    seq = subset[feature_cols].tail(seq_len).values.astype(np.float32)
    return np.expand_dims(seq, axis=0)


def hybrid_predict(tabular, sequence):
    xgb_probs = xgb_model.predict_proba([tabular])[0]
    lstm_probs = lstm_model.predict(sequence, verbose=0)[0]
    return (xgb_probs + lstm_probs) / 2


# -------------------------
# Prediction
# -------------------------
if predict_button and selected_location:

    location_name, lat, lon = selected_location
    lat_grid, lon_grid = get_grid(lat, lon)

    tabular = get_tabular_features(lat_grid, lon_grid)
    sequence = get_sequence(lat_grid, lon_grid)

    if tabular is None or sequence is None:
        st.error("Not enough historical data for prediction")
    else:
        probs = hybrid_predict(tabular, sequence)
        pred = np.argmax(probs)

        labels = ["Low", "Medium", "High"]

        st.session_state.risk = labels[pred]
        st.session_state.prob = probs
        st.session_state.location_name = location_name
        st.session_state.tabular = tabular
        st.session_state.lat = lat
        st.session_state.lon = lon


# -------------------------
# Display Results
# -------------------------
if st.session_state.risk:

    st.success(f"📍 Location: {st.session_state.location_name}")

    st.subheader("🚨 Hybrid Disaster Risk")
    st.success(f"Risk Level: {st.session_state.risk}")

    # Gauge
    risk_value = {"Low": 30, "Medium": 60, "High": 90}[st.session_state.risk]

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_value,
        title={'text': "Risk Level"},
        gauge={'axis': {'range': [0, 100]}}
    ))

    st.plotly_chart(gauge)

    # Probability chart
    prob_df = pd.DataFrame({
        "Risk": ["Low", "Medium", "High"],
        "Probability": st.session_state.prob
    })

    st.plotly_chart(px.bar(prob_df, x="Risk", y="Probability", color="Risk"))

    # -------------------------
    # Feature Importance
    # -------------------------
    st.subheader("🔍 Key Risk Factors")

    importances = xgb_model.feature_importances_

    feature_names = [
        "Earthquake Count", "Magnitude", "Depth",
        "Fire Count", "Brightness", "FRP",
        "Prev Earthquake", "Prev Fire"
    ]

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.plotly_chart(px.bar(fi_df, x="Importance", y="Feature", orientation='h'))

    # -------------------------
    # Trend Chart
    # -------------------------
    st.subheader("📈 Disaster Trend")

    lat_grid, lon_grid = get_grid(st.session_state.lat, st.session_state.lon)

    region = data[
        (data["lat_grid"] == lat_grid) &
        (data["lon_grid"] == lon_grid)
    ]

    if not region.empty:
        trend = region.groupby("year")[["earthquake_count", "fire_count"]].mean().reset_index()

        st.plotly_chart(px.line(
            trend,
            x="year",
            y=["earthquake_count", "fire_count"],
            title="Earthquake vs Wildfire Trend"
        ))

    # -------------------------
    # Risk Explanation
    # -------------------------
    st.subheader("🧠 Why this Risk?")

    tab = st.session_state.tabular

    reasons = []

    if tab[0] > 2:
        reasons.append("High earthquake activity")
    if tab[3] > 2:
        reasons.append("Frequent wildfire activity")
    if tab[6] > 1:
        reasons.append("Recent earthquake history")
    if tab[7] > 1:
        reasons.append("Recent wildfire history")

    if not reasons:
        reasons.append("Low historical disaster activity")

    for r in reasons:
        st.write(f"• {r}")


# -------------------------
# Model Comparison
# -------------------------
st.subheader("📊 Model Performance Comparison")

performance = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost", "LSTM"],
    "Accuracy": [0.58, 0.92, 0.97, 0.65]
})

st.plotly_chart(px.bar(performance, x="Model", y="Accuracy", color="Model"))