# STEP 2: Convert hourly to daily average (easier to forecast), check seasonality
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('electricity_dah_prices.csv')
df['hour_start'] = df['hour'].str.split(' - ').str[0]
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour_start'], format='%Y/%m/%d %H:%M')
df = df.sort_values('datetime').reset_index(drop=True)

# Fill the 1 missing value with interpolation (forward/backward fill)
df['germany'] = df['germany'].interpolate()

# Aggregate hourly -> daily average price
daily = df.set_index('datetime')['germany'].resample('D').mean()
daily = daily.to_frame(name='price')

print(daily.head())
print(f"\nTotal days: {len(daily)}")
print(f"Missing days: {daily['price'].isnull().sum()}")

# Decompose into Trend, Seasonality, Residual
decomposition = seasonal_decompose(daily['price'], model='additive', period=7)  # weekly seasonality

fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.savefig('step2_decomposition.png', dpi=100)
print("Saved plot: step2_decomposition.png")

daily.to_csv('daily_prices.csv')
print("Saved: daily_prices.csv (used in next steps)")
