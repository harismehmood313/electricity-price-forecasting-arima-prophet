# STEP 3b: HYPOTHESIS TESTING
# A Data Scientist doesn't just build a model - they validate assumptions with statistics first.
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

daily = pd.read_csv('daily_prices.csv', index_col='datetime', parse_dates=True)

# Add day-of-week feature
daily['day_of_week'] = daily.index.dayofweek  # 0=Monday, 6=Sunday
daily['is_weekend'] = daily['day_of_week'].isin([5, 6])

weekday_prices = daily[~daily['is_weekend']]['price']
weekend_prices = daily[daily['is_weekend']]['price']

print(f"Weekday average price: EUR {weekday_prices.mean():.2f} (n={len(weekday_prices)})")
print(f"Weekend average price: EUR {weekend_prices.mean():.2f} (n={len(weekend_prices)})")

# HYPOTHESIS TEST: Independent samples t-test
# H0: no difference in mean price between weekday and weekend
# H1: there IS a difference
t_stat, p_value = stats.ttest_ind(weekday_prices, weekend_prices, equal_var=False)

print(f"\n--- Independent t-test ---")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    print("Result: REJECT H0 - weekday and weekend prices ARE significantly different.")
    print("Business implication: day-of-week should be used as a feature in forecasting.")
else:
    print("Result: FAIL TO REJECT H0 - no significant difference detected.")

# Visualise the difference
plt.figure(figsize=(8, 5))
sns.boxplot(data=daily, x='is_weekend', y='price')
plt.xticks([0, 1], ['Weekday', 'Weekend'])
plt.ylabel('Daily Average Price (EUR/MWh)')
plt.title('Electricity Price: Weekday vs Weekend (with t-test)')
plt.savefig('step3b_hypothesis_test.png', dpi=100, bbox_inches='tight')
print("\nSaved plot: step3b_hypothesis_test.png")

daily.to_csv('daily_prices_features.csv')
print("Saved: daily_prices_features.csv (with day_of_week, is_weekend features)")
