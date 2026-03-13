import streamlit as st
import pandas as pd
import joblib
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim

st.set_page_config(page_title="Disaster Risk Prediction", layout="wide")

st.title("🌍 Multi-Hazard Disaster Risk Prediction System")
st.write("Predict disaster risk using earthquake and wildfire indicators.")

# -------------------------
# Load model and datasets
# -------------------------

model = joblib.load("models/best_disaster_model.pkl")

data = pd.read_csv("data/disaster_filtered_classification_dataset.csv")
earthquakes = pd.read_csv("data/earthquakes_clean.csv")
wildfires = pd.read_csv("data/wildfires_clean.csv")

# -------------------------
# Geocoder
# -------------------------

geolocator = Nominatim(user_agent="disaster_app")

# -------------------------
# Session state
# -------------------------

if "lat" not in st.session_state:
    st.session_state.lat = None
    st.session_state.lon = None
    st.session_state.location_name = None
    st.session_state.features = None
    st.session_state.risk = None
    st.session_state.prob = None

# -------------------------
# Sidebar
# -------------------------

st.sidebar.header("Location Selection")

city_query = st.sidebar.text_input(
    "Start typing a city or country",
    placeholder="Tokyo, Delhi, California..."
)

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

predict_button = st.sidebar.button("Predict Risk")

# -------------------------
# Grid conversion
# -------------------------

def get_grid(lat, lon):

    lat_grid = round(lat / 5) * 5
    lon_grid = round(lon / 5) * 5

    return lat_grid, lon_grid


# -------------------------
# Feature extraction
# -------------------------

def get_features(lat_grid, lon_grid):

    subset = data[
        (data["lat_grid"] == lat_grid) &
        (data["lon_grid"] == lon_grid)
    ]

    if subset.empty:
        return None

    row = subset.iloc[0]

    return [
        row["earthquake_count"],
        row["avg_magnitude"],
        row["avg_depth"],
        row["fire_count"],
        row["avg_brightness"],
        row["avg_frp"]
    ]


# -------------------------
# Prediction
# -------------------------

if predict_button and selected_location:

    location_name, lat, lon = selected_location

    st.session_state.lat = lat
    st.session_state.lon = lon
    st.session_state.location_name = location_name

    lat_grid, lon_grid = get_grid(lat, lon)

    features = get_features(lat_grid, lon_grid)

    if features is None:
        st.error("No disaster data available for this region")

    else:

        pred = model.predict([features])[0]
        prob = model.predict_proba([features])[0]

        labels = {0: "Low", 1: "Medium", 2: "High"}

        st.session_state.risk = labels[pred]
        st.session_state.features = features
        st.session_state.prob = prob


# -------------------------
# Display Results
# -------------------------

if st.session_state.risk:

    lat = st.session_state.lat
    lon = st.session_state.lon
    features = st.session_state.features

    lat_grid, lon_grid = get_grid(lat, lon)

    st.success(f"Location Found: {st.session_state.location_name}")

    st.subheader("🚨 Predicted Disaster Risk")
    st.success(f"Risk Level: {st.session_state.risk}")

    # -------------------------
    # Risk Gauge
    # -------------------------

    risk_value = {"Low": 30, "Medium": 60, "High": 90}[st.session_state.risk]

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_value,
        title={'text': "Disaster Risk Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(gauge)

    # -------------------------
    # Probability chart
    # -------------------------

    prob_df = pd.DataFrame({
        "Risk Level": ["Low", "Medium", "High"],
        "Probability": st.session_state.prob
    })

    fig_prob = px.bar(
        prob_df,
        x="Risk Level",
        y="Probability",
        color="Risk Level",
        title="Prediction Probability Distribution"
    )

    st.plotly_chart(fig_prob)

    # -------------------------
    # Hazard indicators
    # -------------------------

    st.subheader("Hazard Indicators")

    col1, col2, col3 = st.columns(3)

    col1.metric("Earthquake Count", features[0])
    col2.metric("Avg Magnitude", round(features[1], 2))
    col3.metric("Wildfire Intensity", round(features[4], 2))

    # -------------------------
    # Hazard comparison chart
    # -------------------------

    hazard_df = pd.DataFrame({
        "Hazard": ["Earthquakes", "Wildfires"],
        "Value": [features[0], features[3]]
    })

    fig_hazard = px.bar(
        hazard_df,
        x="Hazard",
        y="Value",
        color="Hazard",
        title="Hazard Indicators"
    )

    st.plotly_chart(fig_hazard)

    # -------------------------
    # Timeline (FIXED)
    # -------------------------

    st.subheader("📈 Disaster Activity Over Time")

    eq_region = earthquakes[
        (abs(earthquakes["lat_grid"] - lat_grid) <= 5) &
        (abs(earthquakes["lon_grid"] - lon_grid) <= 5)
    ]

    fire_region = wildfires[
        (abs(wildfires["lat_grid"] - lat_grid) <= 5) &
        (abs(wildfires["lon_grid"] - lon_grid) <= 5)
    ]

    eq_timeline = eq_region.groupby("year").size().reset_index(name="earthquakes")
    fire_timeline = fire_region.groupby("year").size().reset_index(name="wildfires")

    timeline = pd.merge(eq_timeline, fire_timeline, on="year", how="outer").fillna(0)

    if not timeline.empty:

        fig_time = px.line(
            timeline,
            x="year",
            y=["earthquakes", "wildfires"],
            title="Earthquake vs Wildfire Activity Over Time"
        )

        st.plotly_chart(fig_time)

    else:
        st.write("No historical activity available.")

    # -------------------------
    # Map
    # -------------------------

    st.subheader("📍 Disaster Risk Map")

    m = folium.Map(location=[lat, lon], zoom_start=4)

    folium.Marker([lat, lon], tooltip="Selected Location").add_to(m)

    heat_data = []

    for _, row in data.iterrows():

        intensity = row["earthquake_count"] + row["fire_count"]

        if intensity > 0:
            heat_data.append([
                row["lat_grid"],
                row["lon_grid"],
                intensity
            ])

    HeatMap(
        heat_data,
        radius=10,
        blur=8,
        min_opacity=0.4
    ).add_to(m)

    st_folium(m, width=900)

    # -------------------------
    # Recent Earthquakes
    # -------------------------

    st.subheader("Recent Earthquakes Near This Location")

    eq_events = eq_region.sort_values("time", ascending=False).head(5)

    if not eq_events.empty:

        st.dataframe(
            eq_events[["time", "latitude", "longitude", "mag"]]
            .rename(columns={
                "time": "Date",
                "mag": "Magnitude"
            })
        )

    else:
        st.write("No recent earthquakes recorded.")

    # -------------------------
    # Recent Wildfires
    # -------------------------

    st.subheader("Recent Wildfires Near This Location")

    fire_events = fire_region.sort_values("acq_date", ascending=False).head(5)

    if not fire_events.empty:

        st.dataframe(
            fire_events[["acq_date", "latitude", "longitude", "brightness", "frp"]]
            .rename(columns={
                "acq_date": "Date",
                "brightness": "Brightness",
                "frp": "Fire Radiative Power"
            })
        )

    else:
        st.write("No recent wildfire events recorded.")

# -------------------------
# Model comparison
# -------------------------

st.subheader("Model Performance Comparison")

performance = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [0.52, 0.39, 0.40]
})

fig_models = px.bar(
    performance,
    x="Model",
    y="Accuracy",
    color="Model",
    title="Model Accuracy Comparison"
)

st.plotly_chart(fig_models)