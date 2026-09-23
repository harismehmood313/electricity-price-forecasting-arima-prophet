# STEP 1: Load and Explore Data
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('electricity_dah_prices.csv')

# Combine date + hour into a single datetime column
df['hour_start'] = df['hour'].str.split(' - ').str[0]  # e.g. "00:00"
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour_start'], format='%Y/%m/%d %H:%M')

# Sort by time (important for time series!)
df = df.sort_values('datetime').reset_index(drop=True)

# Quick look
print(df.head())
print(df.info())
print(df.describe())

# Plot Germany prices over the year to see trend/seasonality
plt.figure(figsize=(14, 5))
plt.plot(df['datetime'], df['germany'])
plt.title('Germany Day-Ahead Electricity Price - 2022 (Hourly)')
plt.xlabel('Date')
plt.ylabel('Price (EUR/MWh)')
plt.savefig('step1_germany_raw.png', dpi=100, bbox_inches='tight')
print("Saved plot: step1_germany_raw.png")
