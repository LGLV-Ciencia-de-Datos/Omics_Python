#!/usr/bin/env python3
# -*- coding: utf-8 -*-
url = 'https://raw.githubusercontent.com/LGLV-Ciencia-de-Datos/Omics_Python/refs/heads/main/Data1.csv'
url2 = 'https://raw.githubusercontent.com/LGLV-Ciencia-de-Datos/Omics_Python/refs/heads/main/Data3.csv'
"""
Created on Wed Oct 29 20:08:01 2025

@author: Eliel
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# ---- 1. Load Data ----
dataset_raw = pd.read_csv(url, index_col=0)
dataset = dataset_raw.copy()
print("Shape:", dataset.shape)
print("First 5 rows:\n", dataset.head())

# ---- 2. Visualize Missing Values ----
plt.figure(figsize=(12, 8))
sns.heatmap(dataset.isnull(), cbar=False, cmap='viridis')
plt.xticks(rotation=45, ha="right")
plt.title('Missing Values Heatmap')
plt.show()

# ---- 3. Mark zeros as NaN ----
dataset.replace(0, np.nan, inplace=True)

# ---- 4. Show missing data summary ----
missing_count = dataset.isnull().sum()
missing_percent = 100 * missing_count / len(dataset)
summary = pd.DataFrame({'MissingCount': missing_count, 'MissingPercent': missing_percent})
print("\nMissing data summary per column:\n", summary)

# ---- 5. Impute missing values ----
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputed_array = imputer.fit_transform(dataset)
ImputedDataMean = pd.DataFrame(imputed_array, columns=dataset.columns, index=dataset.index)

# ---- 6. Confirm no missing left ----
print(f"\nMissing after imputation: {np.isnan(imputed_array).sum()} cells")

# ---- 7. Save imputed data ----
# ImputedDataMean.to_csv('imputed-mean-Data1.csv')
# print("Imputed data saved to 'imputed-mean-Data1.csv'.")

# ---- 8. Compare distributions ----
plt.rcParams['svg.fonttype'] = 'none'
fig, axes = plt.subplots(nrows=2, figsize=(10, 12), sharex=True)
sns.boxplot(ax=axes[0], data=dataset, color='skyblue')
axes[0].set_title('Original Data (after zero→NaN)')
axes[0].tick_params(axis='x', labelrotation=60)
sns.boxplot(ax=axes[1], data=ImputedDataMean, color='salmon')
axes[1].set_title('Imputed Data (Mean)')
axes[1].tick_params(axis='x', labelrotation=60)
plt.tight_layout()
plt.show()

# ---- 9. Side-by-side missing value heatmaps ----
fig, axes = plt.subplots(ncols=2, figsize=(14, 6), sharey=True)
sns.heatmap(dataset.isnull(), ax=axes[0], cbar=False, cmap='viridis')
axes[0].set_title('Missing Values Heatmap')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
sns.heatmap(ImputedDataMean.isnull(), ax=axes[1], cbar=False, cmap='viridis')
axes[1].set_title('Missing After Imputation')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ---- 10. Compare column sums and distributions ----
columns_to_compare = dataset.columns[:6]  # or adjust as needed
columns_sum_raw = dataset[columns_to_compare].sum()
columns_sum_Imputed = ImputedDataMean[columns_to_compare].sum()

plt.figure(figsize=(12, 6))
sns.kdeplot(columns_sum_raw, label='Raw (with NaN)', color='blue', fill=True)
sns.kdeplot(columns_sum_Imputed, label='Imputed (Mean)', color='red', linestyle='--', fill=True)
plt.title("Sum across first 6 columns: Raw vs Imputed")
plt.legend()
plt.show()

# ---- 11. Report summary stats ----
print("\nOriginal data summary (with NaN):\n", dataset.describe())
print("\nImputed data summary:\n", ImputedDataMean.describe())




