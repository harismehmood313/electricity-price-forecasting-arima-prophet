# STEP 5: Compare ARIMA vs Prophet
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

daily = pd.read_csv('daily_prices.csv', index_col='datetime', parse_dates=True)['price']

train = daily.iloc[:-30]
test = daily.iloc[-30:]

# Prophet requires specific column names: 'ds' (date) and 'y' (value)
prophet_train = train.reset_index()
prophet_train.columns = ['ds', 'y']

# Build and fit Prophet model
model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
model.fit(prophet_train)

# Forecast next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
prophet_forecast = forecast.iloc[-30:]['yhat'].values

# Evaluate
mae_prophet = mean_absolute_error(test, prophet_forecast)
rmse_prophet = np.sqrt(mean_squared_error(test, prophet_forecast))

print(f"Prophet Results:")
print(f"MAE: {mae_prophet:.2f}")
print(f"RMSE: {rmse_prophet:.2f}")

print(f"\n--- COMPARISON ---")
print(f"ARIMA(2,1,2):  MAE=127.32, RMSE=154.28")
print(f"Prophet:       MAE={mae_prophet:.2f}, RMSE={rmse_prophet:.2f}")

# Plot comparison
plt.figure(figsize=(12, 5))
plt.plot(test.index, test.values, label='Actual', color='green', linewidth=2)
plt.plot(test.index, prophet_forecast, label='Prophet Forecast', color='blue', linestyle='--')
plt.legend()
plt.title('Prophet Forecast vs Actual - Germany Electricity Price')
plt.savefig('step5_prophet_forecast.png', dpi=100, bbox_inches='tight')
print("Saved plot: step5_prophet_forecast.png")
