# Pearls AQI Predictor

An end-to-end, fully serverless machine learning system that forecasts the **Air Quality Index (AQI) of Karachi for the next 3 days**. The system automatically collects live and historical air quality data, engineers features, trains and compares multiple models, retrains itself on a schedule, and serves predictions through a public interactive dashboard with hazardous-air alerts.

**Live dashboard:** https://pearls-aqi-predictor-khi-ayesha.streamlit.app

---

## Overview

Air pollution is a serious public-health concern in Karachi, where AQI frequently reaches unhealthy levels. This project provides an automated 3-day AQI forecast, similar in spirit to a weather forecast, to help people plan ahead. The entire system runs on free, serverless infrastructure with no manual operation required: data is collected hourly, the model is retrained daily, and the dashboard always serves the latest model.

## Architecture

The system follows an MLOps architecture with four components linked through a central feature store and model registry:

1. **Feature Pipeline** (`feature_pipeline.py`) — fetches live weather and pollutant data hourly from the AQICN API, engineers features, and writes them to the Hopsworks Feature Store.
2. **Training Pipeline** (`training.py`) — reads historical features, experiments with three models (Ridge Regression, Random Forest, LSTM), evaluates each with RMSE/MAE/R², and registers the best model in the Hopsworks Model Registry.
3. **Automation (CI/CD)** — GitHub Actions workflows run the feature pipeline hourly and the training pipeline daily, fully unattended.
4. **Web Dashboard** (`app.py`) — a Streamlit app that loads the latest model and recent features, computes a recursive 3-day forecast, and displays it with health alerts.

**Central hub:** [Hopsworks](https://www.hopsworks.ai/) Feature Store + Model Registry.

## Tech Stack

- **Language:** Python
- **ML:** scikit-learn (Ridge, Random Forest), PyTorch (LSTM)
- **Feature Store & Model Registry:** Hopsworks
- **Data sources:** AQICN API (live), Open-Meteo API (historical backfill)
- **Automation:** GitHub Actions
- **Dashboard:** Streamlit (deployed on Streamlit Community Cloud)
- **Explainability:** SHAP

## Data & Features

- **Resolution:** hourly, for Karachi
- **Variables:** PM2.5, PM10, NO₂, O₃, SO₂, CO, plus temperature, humidity, wind, pressure
- **Dataset:** ~4,300 hourly records (~6 months), backfilled via Open-Meteo
- **AQI derivation:** computed from PM2.5 using the US EPA breakpoint formula
- **Engineered features (41):** lagged AQI values, rolling averages and volatility, lagged meteorological drivers, and cyclical (sine/cosine) time encodings
- **Target:** AQI 24 hours ahead (the 3-day forecast is generated recursively at serving time)

## Models & Results

Three models were trained and compared on a chronological train/test split (24-hour-ahead forecast):

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge Regression | 20.31 | 13.76 | 0.103 |
| **Random Forest** | **19.30** | **14.01** | **0.190** |
| LSTM | 25.99 | 18.90 | -0.465 |

**Random Forest** performed best and is promoted to production. It also beats a naive persistence baseline (R² 0.033), confirming it learns genuine structure. The simpler Random Forest outperformed the LSTM, which overfit on the relatively small dataset. SHAP analysis confirmed PM2.5 as the dominant predictor.

*Note: because the model retrains on continuously growing live data, exact metrics vary between runs; these figures are from the model-development dataset and are reproducible from the training notebook.*

## Repository Structure

```
AQI-predictor/
├── feature_pipeline.py        # Hourly live data collection → Feature Store
├── backfill.py                # Historical data backfill (Open-Meteo)
├── training.py                # Trains 3 models, registers best (automation-ready)
├── app.py                     # Streamlit dashboard (3-day forecast + alerts)
├── test_api.py                # API connection test
├── requirements.txt
├── .github/workflows/
│   ├── feature_pipeline.yml   # Hourly schedule
│   └── training_pipeline.yml  # Daily schedule
├── notebooks/
│   ├── eda.ipynb              # Exploratory data analysis
│   ├── training.ipynb         # Full model experimentation + SHAP
│   └── *.png                  # EDA charts and SHAP plots
└── reports/
    └── milestone8_final_report.pdf   # Detailed final report
```

## Running Locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Set your Hopsworks API key as an environment variable:
   ```
   set HOPSWORKS_API_KEY=your_key_here       # Windows
   export HOPSWORKS_API_KEY=your_key_here     # macOS/Linux
   ```
3. Run a component, for example the dashboard:
   ```
   streamlit run app.py
   ```
   or the training pipeline:
   ```
   python training.py
   ```

## Automation

The two GitHub Actions workflows run on schedule (feature pipeline hourly, training pipeline daily) and can also be triggered manually. API credentials are stored as encrypted GitHub Secrets and are never committed to the repository.

## Dashboard Features

- Current AQI with colour-coded health category
- 3-day forecast (recursive multi-step prediction)
- Automated alert when forecast AQI exceeds the unhealthy threshold
- Interactive chart showing recent trend alongside the forecast

## Author

Ayesha Salahuddin
