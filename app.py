"""
Pearls AQI Predictor - Streamlit Dashboard
-------------------------------------------
Loads the trained model and recent features from the Hopsworks Feature Store
and Model Registry, computes a recursive 3-day-ahead AQI forecast for Karachi,
and displays it on an interactive dashboard with a hazardous-AQI alert.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta
import hopsworks
import joblib

st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌫️", layout="wide")


# ---------- AQI category helper ----------
def aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#009966"
    elif aqi <= 100:
        return "Moderate", "#ffde33"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff9933"
    elif aqi <= 200:
        return "Unhealthy", "#cc0033"
    elif aqi <= 300:
        return "Very Unhealthy", "#660099"
    else:
        return "Hazardous", "#7e0023"


# ---------- Load model + data from Hopsworks (cached) ----------
@st.cache_resource(show_spinner="Connecting to Hopsworks...")
def load_model_and_data():
    api_key = os.environ.get("HOPSWORKS_API_KEY") or st.secrets.get("HOPSWORKS_API_KEY")
    project = hopsworks.login(api_key_value=api_key, project="PredictorAQI")

    # Load model from registry
    mr = project.get_model_registry()
    # Load the LATEST version of the production model so the dashboard always
    # serves the freshest model produced by the daily training pipeline.
    # Fall back to version 1 if the version lookup fails for any reason.
    try:
        all_versions = mr.get_models("aqi_best_model")
        latest_version = max(m.version for m in all_versions)
        model = mr.get_model("aqi_best_model", version=latest_version)
    except Exception:
        model = mr.get_model("aqi_best_model", version=1)
    model_dir = model.download()
    rf = joblib.load(os.path.join(model_dir, "model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    with open(os.path.join(model_dir, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    # Load recent features
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=1)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)

    return rf, scaler, feature_cols, df


# ---------- Feature engineering (same as training) ----------
def build_features(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Ensure numeric types (synthetic rows can become object dtype)
    for c in ["aqi", "pm25", "pm10", "no2", "o3", "so2", "co",
              "temperature", "humidity", "wind", "pressure",
              "hour", "day_of_week", "month", "is_weekend"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    lags = [1, 2, 3, 6, 12, 24, 48, 71, 72, 73, 96, 120, 168]
    for lag in lags:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)
    df["aqi_roll_24"] = df["aqi"].shift(1).rolling(24).mean()
    df["aqi_roll_72"] = df["aqi"].shift(1).rolling(72).mean()
    df["aqi_roll_168"] = df["aqi"].shift(1).rolling(168).mean()
    df["aqi_std_24"] = df["aqi"].shift(1).rolling(24).std()
    for col in ["pm25", "wind", "temperature"]:
        df[f"{col}_lag_24"] = df[col].shift(24)
        df[f"{col}_lag_72"] = df[col].shift(72)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


# ---------- Recursive 3-day forecast ----------
def forecast_3_days(rf, feature_cols, df):
    """
    Predict AQI 24h ahead, then recursively feed predictions back
    to forecast day+1, day+2, day+3.
    """
    work = df.copy()
    predictions = []
    last_time = pd.to_datetime(work["timestamp"].iloc[-1])

    for day in range(1, 4):
        feat = build_features(work)
        latest = feat.iloc[[-1]][feature_cols]
        pred = float(rf.predict(latest)[0])
        pred = max(0, pred)  # AQI can't be negative
        predictions.append({"day": day, "date": last_time + timedelta(days=day), "aqi": round(pred)})

        # Append the prediction as a new synthetic row 24h later, so the next
        # iteration can use it as recent history (recursive forecasting)
        new_time = pd.to_datetime(work["timestamp"].iloc[-1]) + timedelta(hours=24)
        new_row = work.iloc[-1].copy()
        new_row["timestamp"] = new_time
        new_row["aqi"] = pred
        new_row["pm25"] = pred  # pm25 closely tracks aqi
        new_row["hour"] = new_time.hour
        new_row["day_of_week"] = new_time.weekday()
        new_row["month"] = new_time.month
        new_row["is_weekend"] = int(new_time.weekday() >= 5)
        work = pd.concat([work, new_row.to_frame().T], ignore_index=True)

    return predictions


# ---------- Main app ----------
st.title("🌫️ Karachi Air Quality Index Predictor")
st.markdown("Forecasting AQI for the next 3 days using a machine learning model trained on historical air quality and weather data.")

try:
    rf, scaler, feature_cols, df = load_model_and_data()

    current_aqi = float(df["aqi"].iloc[-1])
    cat, color = aqi_category(current_aqi)

    # Current AQI
    st.subheader("Current Air Quality")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            f"<div style='background-color:{color};padding:20px;border-radius:10px;text-align:center;'>"
            f"<h1 style='color:white;margin:0;'>{int(current_aqi)}</h1>"
            f"<p style='color:white;margin:0;'>{cat}</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("Latest reading", f"AQI {int(current_aqi)}", cat)

    # 3-day forecast
    st.subheader("3-Day Forecast")
    preds = forecast_3_days(rf, feature_cols, df)

    cols = st.columns(3)
    for i, p in enumerate(preds):
        pcat, pcolor = aqi_category(p["aqi"])
        with cols[i]:
            st.markdown(
                f"<div style='background-color:{pcolor};padding:15px;border-radius:10px;text-align:center;'>"
                f"<p style='color:white;margin:0;font-size:14px;'>{p['date'].strftime('%a, %b %d')}</p>"
                f"<h2 style='color:white;margin:5px 0;'>{p['aqi']}</h2>"
                f"<p style='color:white;margin:0;font-size:13px;'>{pcat}</p></div>",
                unsafe_allow_html=True,
            )

    # Hazardous alert
    max_pred = max(p["aqi"] for p in preds)
    if max_pred > 150:
        worst = max(preds, key=lambda x: x["aqi"])
        wcat, _ = aqi_category(worst["aqi"])
        st.error(
            f"⚠️ **Health Alert:** AQI is forecast to reach **{worst['aqi']} ({wcat})** "
            f"on {worst['date'].strftime('%A, %b %d')}. Sensitive groups should limit outdoor activity."
        )
    else:
        st.success("✅ No hazardous AQI levels forecast over the next 3 days.")

    # Recent trend + forecast chart
    st.subheader("Recent Trend & Forecast")
    recent = df.tail(72).copy()
    recent["timestamp"] = pd.to_datetime(recent["timestamp"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["timestamp"], y=recent["aqi"],
        mode="lines", name="Recent AQI", line=dict(color="#1f77b4"),
    ))
    fig.add_trace(go.Scatter(
        x=[p["date"] for p in preds], y=[p["aqi"] for p in preds],
        mode="lines+markers", name="Forecast", line=dict(color="#ff7f0e", dash="dash"),
        marker=dict(size=10),
    ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="AQI",
        height=400, hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Forecast uses recursive multi-step prediction: each day's forecast builds on the previous day's, "
        "so accuracy decreases for later days. Model: Random Forest (best of Ridge / Random Forest / LSTM)."
    )

except Exception as e:
    st.error(f"Could not load data or model: {e}")
    st.info("Make sure the HOPSWORKS_API_KEY is set (locally as an environment variable, or in Streamlit secrets when deployed).")
