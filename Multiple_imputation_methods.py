#!/usr/bin/env python3
# -*- coding: utf-8 -*-
url = 'https://raw.githubusercontent.com/LGLV-Ciencia-de-Datos/Omics_Python/refs/heads/main/Data1.csv'
url2 = 'https://raw.githubusercontent.com/LGLV-Ciencia-de-Datos/Omics_Python/refs/heads/main/Data3.csv'
url3 = 'https://raw.githubusercontent.com/LGLV-Ciencia-de-Datos/Omics_Python/refs/heads/main/Tommy_Inferys.csv'
"""
Created on Wed Oct 29 20:44:26 2025

@author: Eliel
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ---- 1. Load Data ----
dataset_raw = pd.read_csv(url3, index_col=0)
dataset = dataset_raw.copy()
print(f"Dataset shape: {dataset.shape}")
print("First 5 rows:\n", dataset.head())

# ---- 2. Mark zeros as NaN ----
dataset.replace(0, np.nan, inplace=True)

# ---- 3. Show missing data summary ----
missing_count = dataset.isnull().sum()
missing_percent = 100 * missing_count / len(dataset)
summary = pd.DataFrame({'MissingCount': missing_count, 'MissingPercent': missing_percent})
print("\nMissing data summary per column:\n", summary)

# ---- 4. Mean Imputation ----
mean_imputer = SimpleImputer(strategy='mean')
imputed_mean = mean_imputer.fit_transform(dataset)
ImputedDataMean = pd.DataFrame(imputed_mean, columns=dataset.columns, index=dataset.index)

# ---- 5. Iterative Imputer (Linear Regression) ----
iter_lr_imputer = IterativeImputer(estimator=LinearRegression(), max_iter=40, tol=0.001, initial_strategy='mean', random_state=42)
imputed_lr = iter_lr_imputer.fit_transform(dataset)
ImputedDataLR = pd.DataFrame(imputed_lr, columns=dataset.columns, index=dataset.index)

# ---- 6. Iterative Imputer (Random Forest) ----
iter_rf_imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=-1), max_iter=10, initial_strategy='mean', random_state=42)
imputed_rf = iter_rf_imputer.fit_transform(dataset)
ImputedDataRF = pd.DataFrame(imputed_rf, columns=dataset.columns, index=dataset.index)

# ---- 7. KNN Imputer (on scaled data for stability) ----
scaler = StandardScaler()
dataset_scaled = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns, index=dataset.index)
knn_imputer = KNNImputer(n_neighbors=4)
imputed_knn_scaled = knn_imputer.fit_transform(dataset_scaled)
imputed_knn = scaler.inverse_transform(imputed_knn_scaled)
ImputedDataKNN = pd.DataFrame(imputed_knn, columns=dataset.columns, index=dataset.index)

# ---- 8. Save all results ----
ImputedDataMean.to_csv('imputed_mean_Data1.csv')
ImputedDataLR.to_csv('imputed_LR_Data1.csv')
ImputedDataRF.to_csv('imputed_RF_Data1.csv')
ImputedDataKNN.to_csv('imputed_KNN_Data1.csv')

print("\nAll imputed datasets saved.")

# ---- 9. Plotting: Boxplots ----
plt.rcParams['svg.fonttype'] = 'none'
fig, axes = plt.subplots(nrows=5, figsize=(10, 26), sharex=True)
sns.boxplot(ax=axes[0], data=dataset, color='skyblue')
axes[0].set_title('Original Data (after zero→NaN)')
axes[0].tick_params(axis='x', labelrotation=60)
sns.boxplot(ax=axes[1], data=ImputedDataMean, color='orange')
axes[1].set_title('Imputed Data (Mean)')
axes[1].tick_params(axis='x', labelrotation=60)
sns.boxplot(ax=axes[2], data=ImputedDataLR, color='pink')
axes[2].set_title('Imputed Data (Iterative Linear Regression)')
axes[2].tick_params(axis='x', labelrotation=60)
sns.boxplot(ax=axes[3], data=ImputedDataRF, color='green')
axes[3].set_title('Imputed Data (Iterative Random Forest)')
axes[3].tick_params(axis='x', labelrotation=60)
sns.boxplot(ax=axes[4], data=ImputedDataKNN, color='red')
axes[4].set_title('Imputed Data (KNN)')
axes[4].tick_params(axis='x', labelrotation=60)
plt.tight_layout()
plt.show()

# ---- 10. Plotting: KDE plot of column sums (first 6 columns) ----
columns_to_compare = dataset.columns[:6]
plt.figure(figsize=(12, 7))
for imputed_data, label, color, ls in [
    (ImputedDataMean, 'Mean', 'orange', '-'),
    (ImputedDataLR, 'Iterative LR', 'pink', '--'),
    (ImputedDataRF, 'Iterative RF', 'green', '-.'),
    (ImputedDataKNN, 'KNN', 'red', ':')
]:
    columns_sum = imputed_data[columns_to_compare].sum()
    sns.kdeplot(columns_sum, label=label, color=color, linestyle=ls, fill=True)
columns_sum_raw = dataset[columns_to_compare].sum()
sns.kdeplot(columns_sum_raw, label='Raw (with NaN)', color='blue', fill=True)
plt.title("Sum across first 6 columns: Raw vs Imputed")
plt.legend()
plt.show()

# ---- 11. Plotting: Side-by-side Heatmaps ----
fig, axes = plt.subplots(ncols=5, figsize=(23, 6), sharey=True)
sns.heatmap(dataset.isnull(), ax=axes[0], cbar=False, cmap='viridis')
axes[0].set_title('Raw (zero→NaN)')
for i, (imputed, title) in enumerate([
    (ImputedDataMean, 'Mean'),
    (ImputedDataLR, 'Iterative LR'),
    (ImputedDataRF, 'Iterative RF'),
    (ImputedDataKNN, 'KNN')
]):
    sns.heatmap(imputed.isnull(), ax=axes[i+1], cbar=False, cmap='viridis')
    axes[i+1].set_title(f'Imputed ({title})')
for ax in axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ---- 12. Summary statistics ----
print("\nOriginal data summary (with NaN):\n", dataset.describe())
print("\nImputed data (Mean):\n", ImputedDataMean.describe())
print("\nImputed data (Iterative Linear Regression):\n", ImputedDataLR.describe())
print("\nImputed data (Iterative Random Forest):\n", ImputedDataRF.describe())
print("\nImputed data (KNN):\n", ImputedDataKNN.describe())













#-----------------Comparion plots










import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---- 1. Load Data ----
dataset = pd.read_csv('Data1.csv', index_col=0)
RF = pd.read_csv('imputed-RF-Data1.csv', index_col=0)
LR = pd.read_csv('imputed_LR_Data1.csv', index_col=0)
KNN = pd.read_csv('imputed_KNN_Data1.csv', index_col=0)
Mean = pd.read_csv('imputed_mean_Data1.csv', index_col=0)

# ---- 2. Compare Sums of First 6 Columns (All Rows) ----
columns_to_compare = dataset.columns[:6]
columns_sum_raw = dataset[columns_to_compare].sum()
columns_sum_RF = RF[columns_to_compare].sum()
columns_sum_LR = LR[columns_to_compare].sum()
columns_sum_KNN = KNN[columns_to_compare].sum()
columns_sum_Mean = Mean[columns_to_compare].sum()

plt.figure(figsize=(12, 6))
sns.kdeplot(columns_sum_raw, label='Raw', color='blue', fill=True, alpha=0.5)
sns.kdeplot(columns_sum_RF, label='RF', color='red', linestyle='--', fill=True, alpha=0.3)
sns.kdeplot(columns_sum_LR, label='LR', color='green', linestyle='-.', fill=True, alpha=0.3)
sns.kdeplot(columns_sum_KNN, label='KNN', color='orange', linestyle=':', fill=True, alpha=0.3)
sns.kdeplot(columns_sum_Mean, label='Mean', color='pink', linestyle='--', fill=True, alpha=0.3)
plt.legend()
plt.title("KDE of Column Sums (All Rows)")
plt.show()


# ---- 3. Evaluate Only Rows with Missing Data in Original ----
rows_with_missing = dataset.isnull().any(axis=1)
columns_sum_raw1 = dataset.loc[rows_with_missing, columns_to_compare].sum()
columns_sum_RF1 = RF.loc[rows_with_missing, columns_to_compare].sum()
columns_sum_LR1 = LR.loc[rows_with_missing, columns_to_compare].sum()
columns_sum_KNN1 = KNN.loc[rows_with_missing, columns_to_compare].sum()
columns_sum_Mean1 = Mean.loc[rows_with_missing, columns_to_compare].sum()

plt.figure(figsize=(12, 6))
sns.kdeplot(columns_sum_raw1, label='Raw', color='blue', fill=True, alpha=0.3)
sns.kdeplot(columns_sum_RF1, label='RF', color='red', linestyle='--', fill=True, alpha=0.3)
sns.kdeplot(columns_sum_LR1, label='LR', color='green', linestyle='-.', fill=True, alpha=0.3)
sns.kdeplot(columns_sum_KNN1, label='KNN', color='orange', linestyle=':', fill=True, alpha=0.3)
sns.kdeplot(columns_sum_Mean1, label='Mean', color='pink', linestyle='--', fill=True, alpha=0.3)
plt.legend()
plt.title("KDE of Column Sums (Rows with Missing in Raw)")
plt.show()







