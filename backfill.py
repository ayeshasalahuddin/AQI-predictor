import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import hopsworks
from dotenv import load_dotenv
import time

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

LAT = 24.8607
LON = 67.0011
CITY = "karachi"


def get_aqi_category(aqi):
    if aqi is None:
        return "unknown"
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


def fetch_weather_data(start_date, end_date):
    """Fetch historical hourly weather data from Open-Meteo."""
    print("Fetching weather data from Open-Meteo...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation",
        "format": "json"
    }

    response = requests.get(url, params=params, timeout=60)
    data = response.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": hourly["time"],
        "temperature": hourly["temperature_2m"],
        "humidity": [float(x) if x is not None else 0.0 for x in hourly["relative_humidity_2m"]],
        "wind": hourly["wind_speed_10m"],
        "pressure": hourly["surface_pressure"],
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Weather data fetched: {len(df)} hourly rows")
    return df


def fetch_pollution_data(start_date, end_date):
    """Fetch pollution data from AQICN for each day in range."""
    print("Fetching pollution data from AQICN...")

    rows = []
    current_date = start_date

    while current_date <= end_date:
        url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()

            if data["status"] == "ok":
                d = data["data"]
                iaqi = d.get("iaqi", {})
                aqi = d.get("aqi", 0)

                for hour in range(24):
                    dt = datetime(
                        current_date.year,
                        current_date.month,
                        current_date.day,
                        hour, 0
                    )
                    row = {
                        "aqi": float(aqi) if aqi else 0.0,
                        "pm25": float(iaqi.get("pm25", {}).get("v", 0) or 0),
                        "pm10": float(iaqi.get("pm10", {}).get("v", 0) or 0),
                        "no2": float(iaqi.get("no2", {}).get("v", 0) or 0),
                        "o3": float(iaqi.get("o3", {}).get("v", 0) or 0),
                        "so2": float(iaqi.get("so2", {}).get("v", 0) or 0),
                        "co": float(iaqi.get("co", {}).get("v", 0) or 0),
                        "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    rows.append(row)

        except Exception as e:
            print(f"  Error for {current_date}: {e}")

        current_date += timedelta(days=1)
        time.sleep(0.3)

    pollution_df = pd.DataFrame(rows)
    print(f"Pollution data fetched: {len(pollution_df)} hourly rows")
    return pollution_df


def combine_and_engineer(pollution_df, weather_df):
    """Merge pollution and weather and engineer features."""
    print("Combining and engineering features...")

    df = pd.merge(pollution_df, weather_df, on="timestamp", how="left")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour.astype("int64")
    df["day_of_week"] = df["timestamp"].dt.weekday.astype("int64")
    df["month"] = df["timestamp"].dt.month.astype("int64")
    df["is_weekend"] = (df["timestamp"].dt.weekday >= 5).astype("int64")
    df["aqi_category"] = df["aqi"].apply(get_aqi_category)
    df["city"] = CITY
    df["humidity"] = df["humidity"].astype(float)
    df = df.fillna(0.0)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df = df[[
        "aqi", "pm25", "pm10", "no2", "o3", "so2", "co",
        "temperature", "humidity", "wind", "pressure",
        "hour", "day_of_week", "month", "is_weekend",
        "aqi_category", "timestamp", "city"
    ]]

    print(f"Final dataset: {len(df)} rows, {len(df.columns)} features")
    return df


def store_to_feature_store(df):
    """Insert all rows into Hopsworks Feature Store."""
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project="PredictorAQI"
    )
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        description="Hourly AQI and weather features for Karachi",
        event_time="timestamp",
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["city", "timestamp"])

    print(f"Inserting {len(df)} rows into Feature Store...")
    fg.insert(df)
    print(f"Successfully inserted {len(df)} rows.")


def run_backfill(months=6):
    print("=" * 50)
    print("AQI Historical Backfill Starting...")
    print("=" * 50)

    end_date = datetime.utcnow().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=months * 30)

    print(f"Date range: {start_date} to {end_date}")

    weather_df = fetch_weather_data(start_date, end_date)
    pollution_df = fetch_pollution_data(start_date, end_date)
    df = combine_and_engineer(pollution_df, weather_df)
    store_to_feature_store(df)

    print("=" * 50)
    print("Backfill complete!")
    print(f"Total rows inserted: {len(df)}")
    print("=" * 50)


if __name__ == "__main__":
    run_backfill(months=6)