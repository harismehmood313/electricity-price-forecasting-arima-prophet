# Electricity Price Forecasting — Full Data Science Workflow

End-to-end data science project forecasting German day-ahead electricity prices: from business problem framing through hypothesis testing to model comparison and stakeholder-ready business recommendations.

## Dataset
Hourly day-ahead electricity prices (EUR/MWh) for 6 European countries (France, Italy, Belgium, Spain, UK, Germany), 2022. Source: Kaggle - European Electricity Day-Ahead Prices.

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

## Methodology
1. **Business framing:** Defined the forecasting question and a testable hypothesis before writing any model code.
2. **Data preparation:** Combined date + hour into a datetime index; resampled hourly prices to daily averages.
3. **Seasonality analysis:** Decomposed the series into trend, weekly seasonality, and residual components.
4. **Stationarity testing:** Augmented Dickey-Fuller test; raw series was non-stationary (p=0.12), resolved via first-order differencing (p=0.0001).
5. **Hypothesis testing:** Independent samples t-test confirmed weekday and weekend prices differ significantly (p<0.001).
6. **Modelling:** Compared three approaches — ARIMA(2,1,2), Facebook Prophet, and a feature-engineered Random Forest (lag features, rolling averages, day-of-week).
7. **Business reporting:** Translated statistical and model results into concrete recommendations.

## Results

| Model | MAE (EUR) | RMSE (EUR) |
|---|---|---|
| ARIMA(2,1,2) | 127.32 | 154.28 |
| Prophet | 161.49 | 185.79 |
| **Random Forest (feature-engineered)** | **47.85** | **57.03** |

The Random Forest model, using yesterday's price and a 7-day rolling average as features, outperformed both classical time series methods by a wide margin — a 62% reduction in MAE versus ARIMA.

## Key Business Findings
- **Weekday vs weekend prices differ significantly** (EUR 256 vs EUR 185/MWh, p<0.001) — energy-flexible operations could reduce costs by ~28% by shifting to weekends.
- **Yesterday's price is the single strongest predictor** (81% feature importance), indicating high day-to-day price persistence.
- A simple "yesterday's price" threshold could serve as an early anomaly-detection signal for trading desks.

## Key Skills Demonstrated
- Business problem framing and hypothesis-driven analysis
- Time series preprocessing (resampling, interpolation, decomposition)
- Statistical hypothesis testing (independent t-test)
- ARIMA model specification, stationarity testing, and evaluation
- Feature engineering (lag features, rolling statistics) and Random Forest regression
- Model benchmarking and translating results into business recommendations
