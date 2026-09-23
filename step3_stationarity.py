# STEP 3: Check Stationarity (ARIMA requires stationary data)
import pandas as pd
from statsmodels.tsa.stattools import adfuller

daily = pd.read_csv('daily_prices.csv', index_col='datetime', parse_dates=True)

def check_stationarity(series, label):
    result = adfuller(series.dropna())
    print(f"\n--- {label} ---")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        print("Result: STATIONARY (good for ARIMA)")
    else:
        print("Result: NOT stationary (needs differencing)")

# Test original series
check_stationarity(daily['price'], "Original Price")

# If not stationary, apply differencing (subtract previous day's value)
daily['price_diff'] = daily['price'].diff()
check_stationarity(daily['price_diff'], "After 1st Differencing")

daily.to_csv('daily_prices_diff.csv')
print("\nSaved: daily_prices_diff.csv")
