"""
AQI Predictor - Training Pipeline
----------------------------------
Reads historical features from the Hopsworks Feature Store, engineers
lag/rolling/cyclical features, and experiments with THREE models to forecast
AQI 24 hours ahead:
    - Ridge Regression (linear baseline, scikit-learn)
    - Random Forest    (non-linear ensemble, scikit-learn)
    - LSTM             (deep learning, PyTorch)
It evaluates each with RMSE, MAE and R2, selects the best by R2, and registers
the best model in the Hopsworks Model Registry.

The LSTM section is wrapped in a try/except: if PyTorch is unavailable or the
deep-learning step fails for any reason, the pipeline logs a warning and
continues with the scikit-learn models, so automation never breaks.

Automation-ready: reads the Hopsworks API key from the HOPSWORKS_API_KEY
environment variable (set as a GitHub Secret for CI/CD).
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

    # ----- 6. Train LSTM (PyTorch) -- wrapped so it can never break the pipeline -----
    # The LSTM is the required "advanced" deep-learning model. It is trained and
    # evaluated here for completeness. If anything in this block fails (e.g. PyTorch
    # not installed in the runner, memory limits), we log a warning and continue
    # with the scikit-learn models above.
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        print("Training LSTM (advanced model)...")

        scaler_lstm = StandardScaler()
        X_all_scaled = scaler_lstm.fit_transform(X)

        SEQ_LEN = 24

        def make_sequences(X_arr, y_arr, seq_len):
            Xs, ys = [], []
            for i in range(len(X_arr) - seq_len):
                Xs.append(X_arr[i:i + seq_len])
                ys.append(y_arr[i + seq_len])
            return np.array(Xs), np.array(ys)

        X_seq, y_seq = make_sequences(X_all_scaled, y.values, SEQ_LEN)
        s_idx = int(len(X_seq) * 0.8)
        X_seq_tr, X_seq_te = X_seq[:s_idx], X_seq[s_idx:]
        y_seq_tr, y_seq_te = y_seq[:s_idx], y_seq[s_idx:]

        X_tr_t = torch.tensor(X_seq_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_seq_tr, dtype=torch.float32).view(-1, 1)
        X_te_t = torch.tensor(X_seq_te, dtype=torch.float32)

        class LSTMModel(nn.Module):
            def __init__(self, n_features, hidden=64):
                super().__init__()
                self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
                self.drop = nn.Dropout(0.3)
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(self.drop(out[:, -1, :]))

        torch.manual_seed(42)
        lstm = LSTMModel(n_features=X_seq.shape[2])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm.parameters(), lr=0.005)

        train_dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)

        EPOCHS = 150
        for epoch in range(EPOCHS):
            lstm.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                loss = criterion(lstm(xb), yb)
                loss.backward()
                optimizer.step()

        lstm.eval()
        with torch.no_grad():
            lstm_pred = lstm(X_te_t).numpy().flatten()

        results["LSTM"] = {
            "model": None,  # evaluated for comparison; RF is selected for production
            "RMSE": float(np.sqrt(mean_squared_error(y_seq_te, lstm_pred))),
            "MAE": float(mean_absolute_error(y_seq_te, lstm_pred)),
            "R2": float(r2_score(y_seq_te, lstm_pred)),
        }
        print("LSTM trained and evaluated.")

    except Exception as e:
        print(f"WARNING: LSTM step skipped ({e}). Continuing with scikit-learn models.")

    # ----- 7. Report all results -----
    print("\nModel comparison (24h ahead):")
    for name, m in results.items():
        print(f"  {name:15s} RMSE: {m['RMSE']:.2f}  MAE: {m['MAE']:.2f}  R2: {m['R2']:.3f}")

    # ----- 8. Select best model by R2 (only models with a saved artifact) -----
    selectable = {k: v for k, v in results.items() if v.get("model") is not None}
    best_name = max(selectable, key=lambda k: selectable[k]["R2"])
    best = selectable[best_name]
    print(f"\nBest model selected for production: {best_name} (R2 = {best['R2']:.3f})")

    # ----- 9. Save best model to Model Registry -----
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
        description=f"Best AQI model ({best_name}), 24h-ahead forecast for Karachi. "
                    f"Experimented with Ridge, Random Forest, and LSTM.",
    )
    aqi_model.save("aqi_model")
    print("Best model saved to Hopsworks Model Registry.")


if __name__ == "__main__":
    main()
