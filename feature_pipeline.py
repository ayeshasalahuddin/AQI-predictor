import os
import tempfile
os.environ["HOPSWORKS_HADOOP_HOME"] = tempfile.gettempdir()
import requests
import pandas as pd
from datetime import datetime
import hopsworks
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
CITY = "karachi"


def fetch_raw_data():
    """Fetch raw AQI and weather data from AQICN API."""
    url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
    response = requests.get(url)
    data = response.json()

    if data["status"] != "ok":
        raise Exception(f"API error: {data}")

    d = data["data"]
    iaqi = d.get("iaqi", {})

    raw = {
        "aqi": d.get("aqi", None),
        "pm25": iaqi.get("pm25", {}).get("v", None),
        "pm10": iaqi.get("pm10", {}).get("v", None),
        "no2": iaqi.get("no2", {}).get("v", None),
        "o3": iaqi.get("o3", {}).get("v", None),
        "so2": iaqi.get("so2", {}).get("v", None),
        "co": iaqi.get("co", {}).get("v", None),
        "temperature": iaqi.get("t", {}).get("v", None),
        "humidity": iaqi.get("h", {}).get("v", None),
        "wind": iaqi.get("w", {}).get("v", None),
        "pressure": iaqi.get("p", {}).get("v", None),
        "timestamp": datetime.utcnow(),
    }

    print(f"Fetched AQI: {raw['aqi']} at {raw['timestamp']}")
    return raw


def engineer_features(raw):
    """Compute engineered features from raw data."""
    now = raw["timestamp"]

    features = {
        # Raw values
        "aqi": float(raw["aqi"]) if raw["aqi"] is not None else 0.0,
        "pm25": float(raw["pm25"]) if raw["pm25"] is not None else 0.0,
        "pm10": float(raw["pm10"]) if raw["pm10"] is not None else 0.0,
        "no2": float(raw["no2"]) if raw["no2"] is not None else 0.0,
        "o3": float(raw["o3"]) if raw["o3"] is not None else 0.0,
        "so2": float(raw["so2"]) if raw["so2"] is not None else 0.0,
        "co": float(raw["co"]) if raw["co"] is not None else 0.0,
        "temperature": float(raw["temperature"]) if raw["temperature"] is not None else 0.0,
        "humidity": float(raw["humidity"]) if raw["humidity"] is not None else 0.0,
        "wind": float(raw["wind"]) if raw["wind"] is not None else 0.0,
        "pressure": float(raw["pressure"]) if raw["pressure"] is not None else 0.0,

        # Time-based features
        "hour": int(now.hour),
        "day_of_week": int(now.weekday()),   # 0=Monday, 6=Sunday
        "month": int(now.month),
        "is_weekend": int(now.weekday() >= 5),

        # AQI category (for reference)
        "aqi_category": get_aqi_category(raw["aqi"]),

        # Timestamp
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "city": CITY,
    }

    print("Engineered features:")
    for k, v in features.items():
        print(f"  {k}: {v}")

    return features


def get_aqi_category(aqi):
    """Return AQI health category as a string."""
    if aqi is None:
        return "unknown"
    aqi = int(aqi)
    if aqi <= 50:
        return "good"
    elif aqi <= 100:
        return "moderate"
    elif aqi <= 150:
        return "unhealthy_sensitive"
    elif aqi <= 200:
        return "unhealthy"
    elif aqi <= 300:
        return "very_unhealthy"
    else:
        return "hazardous"


def store_to_feature_store(features):
    """Connect to Hopsworks and insert features into the Feature Store."""
    print("\nConnecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project="PredictorAQI")
    fs = project.get_feature_store()

    # Create a dataframe with one row
    df = pd.DataFrame([features])

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Connected. Inserting into Feature Store...")

    # Get or create feature group
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        description="Hourly AQI and weather features for Karachi",
        event_time="timestamp",
    )

    fg.insert(df)
    print(f"Successfully inserted row into Feature Store.")
    print(f"Feature group: aqi_features | Rows inserted: 1")


def run_pipeline():
    """Run the full feature pipeline."""
    print("=" * 50)
    print("AQI Feature Pipeline Starting...")
    print("=" * 50)

    raw = fetch_raw_data()
    features = engineer_features(raw)
    store_to_feature_store(features)

    print("=" * 50)
    print("Pipeline complete.")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()