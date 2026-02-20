


#testing normality

####D’Agostino-Pearson (normaltest)


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import normaltest, anderson
from statannotations.Annotator import Annotator
from scipy import stats


# Load dataset
dataset = pd.read_csv('MinMaxScaler_intensities.csv', index_col=0)

dataset

dataset.columns


# Select the column you wantto test (example: '1cm_1')
col_name = 'Time0_1'

data = dataset[col_name].dropna()  # remove NaNs

# --- Normality Test: D'Agostino-Pearson ---
stat, p_value = normaltest(data)
print(f"D'Agostino-Pearson Test Statistic: {stat:.3f}")
print(f"P-value: {p_value:.3f}")



alpha = 0.05

if p_value > alpha:
    print("Fail to reject the null hypothesis: Data appears to be normally distributed.")
else:
    print("Reject the null hypothesis: Data does not appear to be normally distributed.")


# --- Optional: Anderson-Darling test ---
result = anderson(data)


print("\nAnderson-Darling Test Statistic:", result.statistic)
for sl, crit in zip(result.significance_level, result.critical_values):
    print(f"Significance Level: {sl}%  Critical Value: {crit}")
if result.statistic < result.critical_values[2]:
    print("Data appears normal at 5% significance level.")
else:
    print("Data does NOT appear normal at 5% significance level.")



# --- Plot distribution ---
plt.figure(figsize=(12, 6))
sns.histplot(data, bins=50, kde=True, color='blue', alpha=0.6)
plt.title(f'Distribution of {col_name}', fontsize=16)
plt.xlabel(col_name, fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()
