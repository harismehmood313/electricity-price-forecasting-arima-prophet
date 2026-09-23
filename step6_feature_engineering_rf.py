# STEP 6: FEATURE-ENGINEERED MODEL (Random Forest) - using the hypothesis test insight
# This shows the ML side of a Data Scientist's toolkit, not just classical time series.
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

daily = pd.read_csv('daily_prices_features.csv', index_col='datetime', parse_dates=True)

# FEATURE ENGINEERING - using insight from Step 3b (day-of-week matters)
daily['lag_1'] = daily['price'].shift(1)     # yesterday's price
daily['lag_7'] = daily['price'].shift(7)     # same day last week
daily['rolling_mean_7'] = daily['price'].shift(1).rolling(7).mean()  # last week's average
daily['month'] = daily.index.month

daily = daily.dropna()

features = ['day_of_week', 'is_weekend', 'lag_1', 'lag_7', 'rolling_mean_7', 'month']
X = daily[features]
y = daily['price']

# Time-based split (never shuffle time series data)
split_point = len(daily) - 30
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Random Forest (feature-engineered) Results:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Feature importance - which factors actually drive price?
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print(f"\nFeature Importance:")
print(importance)

plt.figure(figsize=(8, 5))
importance.plot(kind='barh')
plt.title('What Drives Electricity Price? (Random Forest Feature Importance)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('step6_feature_importance.png', dpi=100, bbox_inches='tight')
print("\nSaved plot: step6_feature_importance.png")

print(f"\n--- FINAL MODEL COMPARISON ---")
print(f"ARIMA(2,1,2):    MAE=127.32, RMSE=154.28")
print(f"Prophet:         MAE=161.49, RMSE=185.79")
print(f"Random Forest:   MAE={mae:.2f}, RMSE={rmse:.2f}")
