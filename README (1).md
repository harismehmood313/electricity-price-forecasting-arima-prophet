# Electricity Price Forecasting — ARIMA vs Prophet vs Random Forest ⚡📈

End-to-end data science project forecasting German day-ahead electricity prices — from business problem framing and hypothesis testing to model comparison and stakeholder-ready business recommendations.

## Overview

This project analyses **8,762 hourly day-ahead electricity price records** (2022) across 6 European countries, sourced from Kaggle's European Electricity Day-Ahead Prices dataset. The focus is Germany's price series.

The goal was to test a statistical hypothesis about price behaviour, build and benchmark three different forecasting approaches, and translate the results into a concrete business recommendation.

## What This Project Covers

- **Business Framing** — defined the forecasting question and a testable hypothesis before writing any model code
- **Data Preparation** — combined date and hour into a datetime index, resampled hourly prices to daily averages
- **Seasonality Analysis** — decomposed the series into trend, weekly seasonality, and residual components
- **Stationarity Testing** — Augmented Dickey-Fuller test, resolved non-stationarity via first-order differencing
- **Hypothesis Testing** — independent samples t-test comparing weekday vs weekend prices
- **Modelling** — ARIMA(2,1,2), Facebook Prophet, and a feature-engineered Random Forest (lag features, rolling averages, day-of-week)
- **Business Reporting** — translated statistical and model results into actionable recommendations

## Key Findings

| Metric | Value |
|---|---|
| Records analysed | 8,762 (hourly), 365 (daily) |
| Weekday average price | €256.00/MWh |
| Weekend average price | €184.59/MWh |
| Weekday vs weekend difference | Statistically significant (p < 0.001) |
| Best model | Random Forest (feature-engineered) |
| Best model MAE | €47.85 |

- **Weekday and weekend prices differ significantly** — not random noise — suggesting energy-flexible operations could cut costs by shifting to weekends.
- **A feature-engineered Random Forest outperformed both classical time series models by a wide margin**, driven primarily by yesterday's price (81% feature importance) — showing electricity prices are highly persistent day-to-day.
- ARIMA outperformed Prophet on this dataset, likely because Prophet is tuned for longer seasonal/holiday patterns rather than strong short-term autocorrelation.

## Visualisations

**Raw Price Series (Germany, 2022)**

[![Germany Price Series](https://github.com/harismehmood313/electricity-price-forecasting-arima-prophet/raw/main/step1_germany_raw.png)](https://github.com/harismehmood313/electricity-price-forecasting-arima-prophet/blob/main/step1_germany_raw.png)

**Seasonal Decomposition (Trend, Weekly Seasonality, Residual)**

[![Seasonal Decomposition](https://github.com/harismehmood313/electricity-price-forecasting-arima-prophet/raw/main/step2_decomposition.png)](https://github.com/harismehmood313/electricity-price-forecasting-arima-prophet/blob/main/step2_decomposition.png)

**Hypothesis Test: Weekday vs Weekend Prices**

[![Hypothesis Test](https://github.com/harismehmood313/electricity-price-forecasting-arima-prophet/raw/main/step3b_hypothesis_test.png)](https://github.com/harismehmood313/electricity-price-forecasting-arima-prophet/blob/main/step3b_hypothesis_test.png)

**ARIMA Forecast vs Actual**

[![ARIMA Forecast](https://github.com/harismehmood313/electricity-price-forecasting-arima-prophet/raw/main/step4_arima_forecast.png)](https://github.com/harismehmood313/electricity-price-forecasting-arima-prophet/blob/main/step4_arima_forecast.png)

**Prophet Forecast vs Actual**

[![Prophet Forecast](https://github.com/harismehmood313/electricity-price-forecasting-arima-prophet/raw/main/step5_prophet_forecast.png)](https://github.com/harismehmood313/electricity-price-forecasting-arima-prophet/blob/main/step5_prophet_forecast.png)

**Random Forest Feature Importance**

[![Feature Importance](https://github.com/harismehmood313/electricity-price-forecasting-arima-prophet/raw/main/step6_feature_importance.png)](https://github.com/harismehmood313/electricity-price-forecasting-arima-prophet/blob/main/step6_feature_importance.png)

## Results

| Model | MAE (EUR) | RMSE (EUR) |
|---|---|---|
| ARIMA(2,1,2) | 127.32 | 154.28 |
| Prophet | 161.49 | 185.79 |
| **Random Forest (feature-engineered)** | **47.85** | **57.03** |

## Tech Stack

- **Python**
- **Pandas** & **NumPy** — data preparation
- **Statsmodels** — ARIMA, seasonal decomposition, stationarity testing
- **Prophet** — automated time series forecasting
- **Scikit-learn** — Random Forest Regressor
- **SciPy** — hypothesis testing
- **Matplotlib** & **Seaborn** — visualisation

## Project Structure

```
step0_business_problem.py       # Business question and hypothesis framing
step1_load_explore.py           # Load data, build datetime index, initial visualisation
step2_daily_seasonality.py      # Aggregate hourly -> daily, seasonal decomposition
step3_stationarity.py           # ADF test for stationarity, differencing
step3b_hypothesis_testing.py    # Statistical t-test: weekday vs weekend prices
step4_arima_model.py            # Train/test split, ARIMA(2,1,2) model, forecast
step5_prophet_comparison.py     # Prophet model, compare against ARIMA
step6_feature_engineering_rf.py # Feature engineering + Random Forest benchmark
step7_business_insights.py      # Executive summary and business recommendations
electricity_dah_prices.csv      # Raw dataset
requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt
python step1_load_explore.py
python step2_daily_seasonality.py
python step3_stationarity.py
python step3b_hypothesis_testing.py
python step4_arima_model.py
python step5_prophet_comparison.py
python step6_feature_engineering_rf.py
python step7_business_insights.py
```
Run in order — each step saves an output file used by later steps.

## Data Source

Kaggle (n.d.) *European Electricity Day-Ahead Prices*. Available at: <https://www.kaggle.com/>

## Author

**Haris Mehmood** — MSc Data Analytics, Dublin Business School
