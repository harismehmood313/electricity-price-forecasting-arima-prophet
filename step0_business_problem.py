"""
STEP 0: BUSINESS PROBLEM FRAMING
==================================
This is what a Data Scientist does FIRST, before touching code.

BUSINESS PROBLEM:
Energy trading desks and industrial consumers need to plan purchasing decisions
a day ahead. If we can forecast tomorrow's electricity price with reasonable
accuracy, a business can:
  - Shift energy-intensive operations to cheaper forecasted hours/days
  - Hedge purchasing decisions to avoid price spikes
  - Benchmark whether the market price is "normal" or anomalous

QUESTION WE ARE ANSWERING:
"Can we forecast Germany's next-day electricity price using historical
day-ahead price patterns, and is the pattern different on weekdays vs weekends?"

HYPOTHESIS (to test with statistics, before modelling):
H0 (null): There is NO significant difference between weekday and weekend
           electricity prices.
H1 (alt):  There IS a significant difference between weekday and weekend
           electricity prices.

Why this matters: if weekday/weekend prices differ significantly, that's a
strong signal that "day of week" is a real driver of price and should be
used as a feature in modelling - not just left for ARIMA to infer implicitly.
"""
print(__doc__)
