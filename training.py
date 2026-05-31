"""
AQI Predictor - Training Pipeline
----------------------------------
Reads historical features from the Hopsworks Feature Store, engineers
lag/rolling/cyclical features, trains Ridge Regression and Random Forest
to forecast AQI 24 hours ahead, selects the best model by R2, and registers
it in the Hopsworks Model Registry.

This script is automation-ready: it reads the Hopsworks API key from the
HOPSWORKS_API_KEY environment variable (set as a GitHub Secret for CI/CD).
"""

import os
import json
import numpy as np
import pandas as pd
import hopsworks
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from hsml.schema import Schema
from hsml.model_schema import ModelSchema


def main():
    # ----- 1. Connect to Hopsworks -----
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY environment variable not set.")

    project = hopsworks.login(api_key_value=api_key, project="PredictorAQI")
    fs = project.get_feature_store()

    # ----- 2. Read data from Feature Store -----
    fg = fs.get_feature_group(name="aqi_features", version=1)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {df.shape[0]} rows from Feature Store.")

    # ----- 3. Feature engineering -----
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

    # Target: AQI 24 hours ahead
    df["target_aqi"] = df["aqi"].shift(-24)
    df_model = df.dropna().reset_index(drop=True)

    exclude = ["timestamp", "city", "aqi_category", "target_aqi", "aqi"]
    feature_cols = [c for c in df_model.columns if c not in exclude]

    X = df_model[feature_cols]
    y = df_model["target_aqi"]

    # Time-based split (no shuffling for time series)
    split_idx = int(len(df_model) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows, Features: {X_train.shape[1]}")

    results = {}

    # ----- 4. Train Ridge Regression -----
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_s, y_train)
    rp = ridge.predict(X_test_s)
    results["Ridge"] = {
        "model": ridge,
        "RMSE": float(np.sqrt(mean_squared_error(y_test, rp))),
        "MAE": float(mean_absolute_error(y_test, rp)),
        "R2": float(r2_score(y_test, rp)),
    }

    # ----- 5. Train Random Forest -----
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    fp = rf.predict(X_test)
    results["RandomForest"] = {
        "model": rf,
        "RMSE": float(np.sqrt(mean_squared_error(y_test, fp))),
        "MAE": float(mean_absolute_error(y_test, fp)),
        "R2": float(r2_score(y_test, fp)),
    }

    print("\nModel comparison (24h ahead):")
    for name, m in results.items():
        print(f"  {name:15s} RMSE: {m['RMSE']:.2f}  MAE: {m['MAE']:.2f}  R2: {m['R2']:.3f}")

    # ----- 6. Select best model by R2 -----
    best_name = max(results, key=lambda k: results[k]["R2"])
    best = results[best_name]
    print(f"\nBest model: {best_name} (R2 = {best['R2']:.3f})")

    # ----- 7. Save best model to Model Registry -----
    os.makedirs("aqi_model", exist_ok=True)
    joblib.dump(best["model"], "aqi_model/model.pkl")
    joblib.dump(scaler, "aqi_model/scaler.pkl")
    with open("aqi_model/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    mr = project.get_model_registry()
    aqi_model = mr.python.create_model(
        name="aqi_best_model",
        metrics={"rmse": best["RMSE"], "mae": best["MAE"], "r2": best["R2"]},
        model_schema=model_schema,
        description=f"Best AQI model ({best_name}), 24h-ahead forecast for Karachi.",
    )
    aqi_model.save("aqi_model")
    print("Best model saved to Hopsworks Model Registry.")


if __name__ == "__main__":
    main()
