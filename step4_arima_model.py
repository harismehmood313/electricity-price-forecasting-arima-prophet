# STEP 4: Train/Test Split + ARIMA Model
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

daily = pd.read_csv('daily_prices.csv', index_col='datetime', parse_dates=True)['price']

# Time series split: NEVER shuffle - always split by time (last 30 days = test)
train = daily.iloc[:-30]
test = daily.iloc[-30:]

print(f"Train size: {len(train)} days")
print(f"Test size: {len(test)} days")

# Build ARIMA model: order=(p, d, q)
# d=1 from Step 3 (stationarity). p, q chosen via ACF/PACF or trial (here: common starting point)
model = ARIMA(train, order=(2, 1, 2))
fitted_model = model.fit()
print(fitted_model.summary())

# Forecast next 30 days
forecast = fitted_model.forecast(steps=30)

# Evaluate
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"\nARIMA(2,1,2) Results:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Plot actual vs forecast
plt.figure(figsize=(12, 5))
plt.plot(train.index[-60:], train.values[-60:], label='Train (last 60 days)')
plt.plot(test.index, test.values, label='Actual', color='green')
plt.plot(test.index, forecast, label='ARIMA Forecast', color='red', linestyle='--')
plt.legend()
plt.title('ARIMA Forecast vs Actual - Germany Electricity Price')
plt.savefig('step4_arima_forecast.png', dpi=100, bbox_inches='tight')
print("Saved plot: step4_arima_forecast.png")
