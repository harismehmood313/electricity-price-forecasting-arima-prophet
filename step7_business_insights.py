# STEP 7: BUSINESS INSIGHTS & STAKEHOLDER REPORTING
# This is the step most tutorials skip - but it's the actual point of the job.
import pandas as pd
import matplotlib.pyplot as plt

daily = pd.read_csv('daily_prices_features.csv', index_col='datetime', parse_dates=True)

print("="*60)
print("EXECUTIVE SUMMARY - Electricity Price Forecasting Project")
print("="*60)

print("""
BUSINESS QUESTION:
Can we forecast next-day German electricity prices to support
purchasing and operational decisions?

KEY FINDINGS:

1. WEEKDAY vs WEEKEND PRICING (statistically confirmed):
   - Weekday average: EUR 256.00/MWh
   - Weekend average: EUR 184.59/MWh
   - This EUR 71/MWh gap is statistically significant (p < 0.001),
     not random noise.
   -> RECOMMENDATION: Energy-flexible operations should be scheduled
      on weekends where possible to reduce costs by ~28%.

2. MODEL PERFORMANCE (which approach should the business trust?):
   - ARIMA (classical time series):        MAE EUR 127.32
   - Prophet (automated forecasting tool): MAE EUR 161.49
   - Random Forest (feature-engineered):   MAE EUR 47.85  <- BEST

   -> RECOMMENDATION: Use the Random Forest model for operational
      forecasting - it is 62% more accurate than ARIMA, because it
      captures yesterday's price (lag_1) as the dominant signal
      rather than relying purely on long-run autocorrelation structure.

3. WHAT ACTUALLY DRIVES PRICE:
   - Yesterday's price (lag_1) explains ~81% of the model's predictive
     power. This means electricity prices are highly persistent
     day-to-day - sudden large jumps are the exception, not the norm.
   -> RECOMMENDATION: A simple "yesterday's price" alert threshold
      could flag anomalous trading days for manual review.

LIMITATIONS & NEXT STEPS:
- Only 1 year of data (2022) - a full multi-year dataset would allow
  testing for annual seasonality (e.g., winter vs summer demand).
- External drivers (gas prices, weather, renewable output) were not
  included - adding them would likely improve Random Forest further.
""")
