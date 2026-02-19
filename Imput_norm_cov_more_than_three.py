#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 22:08:48 2026

@author: Eliel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, QuantileTransformer,
    RobustScaler, PowerTransformer
)

# --------------------------
# Load dataset
# --------------------------
df = pd.read_csv("Data5.csv", index_col=0)
df

print("Original shape:", df.shape)

# --------------------------
# Select only the intensity columns
# --------------------------


#  r'^\d+[A-Za-z]+_\d+$'_______________1cm_1', '1cm_2', '1cm_3',

#   r'^[A-Za-z]+\d+_\d+$'________#Time0_1, Dose5_2, Group12_3, Visit1_1


#  '_[0-9]$', ''   ____________  works for both 


df.columns

intensity_cols = df.columns[
    df.columns.str.contains(r'^\d+[A-Za-z]+_\d+$', regex=True)
].tolist()                                                       #Time0_1, Dose5_2, Group12_3, Visit1_1






print(intensity_cols)


X = df[intensity_cols].copy()

# --------------------------
# Treat 0 as missing
# --------------------------


# --------------------------
# Visualize missing values
# --------------------------
plt.figure(figsize=(8, 4))
ax = sns.heatmap(X.isnull(), cbar=False, cmap="viridis")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.title("Missing Values Heatmap (Intensities)")
plt.show()


# --------------------------
# Imputation with IterativeImputer + RandomForest
# --------------------------




from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer, SimpleImputer


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def impute_data(X, method="rf", random_state=42, max_iter=50, tol=1e-2):
    """
    Impute missing values using different methods.

    Parameters
    ----------
    X : DataFrame
        Input data with missing values
    method : str
        'rf', 'knn', 'lr', 'mean'
    """

    method = method.lower()

    if method == "rf":
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(random_state=random_state),
            max_iter=max_iter,
            tol=tol,
            initial_strategy="mean"
        )

    elif method == "lr":
        imputer = IterativeImputer(
            estimator=LinearRegression(),
            max_iter=max_iter,
            tol=tol,
            initial_strategy="mean"
        )

    elif method == "knn":
        imputer = KNNImputer(
            n_neighbors=5,
            weights="distance"
        )

    elif method == "mean":
        imputer = SimpleImputer(strategy="mean")

    else:
        raise ValueError("Method must be: 'rf', 'lr', 'knn', or 'mean'")

    imputed_array = imputer.fit_transform(X)

    return pd.DataFrame(
        imputed_array,
        columns=X.columns,
        index=X.index
    )



#RF
Imputed_RF = impute_data(X, method="rf")
Imputed_RF.to_csv("imputed_RF.csv")

#KNN
Imputed_KNN = impute_data(X, method="knn")
Imputed_KNN.to_csv("imputed_KNN.csv")

#LR
Imputed_LR = impute_data(X, method="lr")
Imputed_LR.to_csv("imputed_LR.csv")

#Mean
Imputed_Mean = impute_data(X, method="mean")
Imputed_Mean.to_csv("imputed_Mean.csv")


#-----------------------------------


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
    PowerTransformer
)

# --------------------------
# Output folder
# --------------------------

output_dir = "results_"
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Load imputed intensity data
# --------------------------

ImputedData = pd.read_csv("imputed_mean.csv", index_col=0)

print("Loaded shape:", ImputedData.shape)


# --------------------------
# Define intensity groups
# (EDIT this to match your columns)
# --------------------------

#groups = {
#    "T0": ["Time0_1", "Time0_2", "Time0_3"],
#    "T6": ["Time6_1", "Time6_2", "Time6_3"]
#}




groups = {
    "1cm": ["1cm_1", "1cm_2", "1cm_3", "1cm_4"],
    "2cm": ["2cm_1", "2cm_2", "2cm_3", "2cm_4"],
    "3cm": ["3cm_1", "3cm_2", "3cm_3", "3cm_4"],
    "4cm": ["4cm_1", "4cm_2", "4cm_3", "4cm_4"],
    "5cm": ["5cm_1", "5cm_2", "5cm_3", "5cm_4"],
}








intensity_cols = ImputedData.columns.tolist()


# --------------------------
# Function: Apply transformation
# --------------------------

def apply_transform(name, transformer, data, columns, save=True):

    transformed = transformer.fit_transform(data)

    result = pd.DataFrame(
        transformed,
        columns=columns,
        index=data.index
    )

    # Ensure positive values
    min_val = result.min().min()

    if min_val <= 0:
        shift = abs(min_val) + 1e-3
        result += shift
        print(f"[{name}] Shift applied: {shift}")

    if save:
        result.to_csv(
            os.path.join(output_dir, f"{name}_intensities.csv")
        )

    return result


# --------------------------
# Define scalers
# --------------------------

scalers = {

    "StandardScaler": StandardScaler(),

    "MinMaxScaler": MinMaxScaler(feature_range=(1, 2)),

    "QuantileTransformer": QuantileTransformer(
        n_quantiles=min(1000, ImputedData.shape[0]),
        output_distribution="normal",
        random_state=0),

    "RobustScaler": RobustScaler(),

    "PowerTrans_YJ": PowerTransformer(method="yeo-johnson"),

    "PowerTrans_BoxCox": PowerTransformer(method="box-cox"),
}


# --------------------------
# Apply transformations
# --------------------------

transformed_dfs = {
    "Imputed": ImputedData
}

print("\nApplying transformations...")

for name, scaler in scalers.items():

    try:

        df_trans = apply_transform(
            name,
            scaler,
            ImputedData,
            intensity_cols
        )

        transformed_dfs[name] = df_trans

        print(f"Saved: {name}")

    except ValueError as e:

        print(f"Skipping {name}: {e}")


# --------------------------
# SQRT transform
# --------------------------

X_sqrt = np.sqrt(ImputedData)

min_val = X_sqrt.min().min()

if min_val <= 0:
    shift = abs(min_val) + 1e-3
    X_sqrt += shift
    print(f"[SQRT] Shift applied: {shift}")

X_sqrt.to_csv(
    os.path.join(output_dir, "SQRT_intensities.csv")
)

transformed_dfs["SQRT"] = X_sqrt


# --------------------------
# Function: Compute CV and mean
# --------------------------

def compute_group_cv(df, groups):

    df_cv = df.copy()

    for group_name, cols in groups.items():

        values = df[cols].to_numpy()

        means = values.mean(axis=1)
        stds = values.std(axis=1)

        cv = np.where(
            means != 0,
            (stds / means) * 100,
            np.nan
        )

        df_cv[f"cv_{group_name}"] = cv
        df_cv[f"mean_{group_name}"] = means

    return df_cv


# --------------------------
# Compute CVs
# --------------------------

df_cv_dict = {}

print("\nComputing CVs...")

for name, df_trans in transformed_dfs.items():

    df_cv = compute_group_cv(df_trans, groups)

    df_cv_dict[name] = df_cv

    path = os.path.join(
        output_dir,
        f"CV_Mean_{name}.csv"
    )

    df_cv.to_csv(path)

    print(f"Saved: {path}")


# --------------------------
# Convert to long format
# --------------------------

df_long = pd.DataFrame(
    columns=["Method", "Group", "CV (%)"]
)

for name, df_cv in df_cv_dict.items():

    for group_name in groups.keys():

        df_tmp = pd.DataFrame({
            "Method": name,
            "Group": group_name,
            "CV (%)": df_cv[f"cv_{group_name}"]
        })

        df_long = pd.concat(
            [df_long, df_tmp],
            ignore_index=True
        )


# --------------------------
# Plot CV distributions
# --------------------------

plt.figure(figsize=(14, 10))

sns.stripplot(
    x="CV (%)",
    y="Method",
    hue="Group",
    data=df_long,
    dodge=True,
    size=5,
    jitter=True,
    alpha=0.7
)

plt.xlabel("Coefficient of Variation (%)", fontsize=14)
plt.ylabel("")
plt.title(
    "CV Distribution Across Groups (All Transformations)",
    fontsize=16
)

plt.legend(title="Group")

plt.tight_layout()
plt.show()











































# --------------------------
# Load imputed intensity data
# --------------------------
ImputedData = pd.read_csv("imputed_RF.csv", index_col=0)

# --------------------------
# Function: Apply transformation and ensure positivity
# --------------------------
def apply_transform(name, transformer, data, columns, save=True):
    transformed = transformer.fit_transform(data)
    result = pd.DataFrame(transformed, columns=columns, index=data.index)
    
    # Make all positive if necessary
    min_val = result.min().min()
    if min_val <= 0:
        shift = abs(min_val) + 1e-3
        result += shift
        print(f"[{name}] Shift applied to make positive: {shift}")
    
    if save:
        result.to_csv(os.path.join(output_dir, f"{name}_intensities.csv"))
    return result

# --------------------------
# Define scalers / transformations
# --------------------------
scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(feature_range=(1, 2)),
    "QuantileTransformer": QuantileTransformer(
        n_quantiles=min(1000, ImputedDataRF.shape[0]),
        output_distribution="normal",
        random_state=0),
    "RobustScaler": RobustScaler(),
    "PowerTrans_YJ": PowerTransformer(method="yeo-johnson"),
    "PowerTrans_BoxCox": PowerTransformer(method="box-cox"),
}

# --------------------------
# Apply transformations
# --------------------------
transformed_dfs = {"Imputed": ImputedData}

for name, scaler in scalers.items():
    try:
        transformed_dfs[name] = apply_transform(name, scaler, ImputedData, intensity_cols)
    except ValueError as e:
        print(f"Skipping {name}: {e}")

# SQRT separately
X_sqrt = np.sqrt(ImputedData)
min_val = X_sqrt.min().min()
if min_val <= 0:
    shift = abs(min_val) + 1e-3
    X_sqrt += shift
    print(f"[SQRT] Shift applied to make positive: {shift}")

X_sqrt.to_csv(os.path.join(output_dir, "SQRT_intensities.csv"))
transformed_dfs["SQRT"] = X_sqrt

# --------------------------
# Function: Compute row-wise CV and mean per group
# --------------------------
def compute_group_cv(df, groups):
    df_cv = df.copy()
    for group_name, cols in groups.items():
        values = df[cols].to_numpy()
        means = values.mean(axis=1)
        stds = values.std(axis=1)
        cv = np.where(means != 0, (stds / means) * 100, np.nan)
        
        df_cv[f"cv_{group_name}"] = cv
        df_cv[f"mean_{group_name}"] = means
    return df_cv

# --------------------------
# Compute CVs for all transformations
# --------------------------
df_cv_dict = {}
for name, df_trans in transformed_dfs.items():
    df_cv = compute_group_cv(df_trans, groups)
    df_cv_dict[name] = df_cv
    # Save CSV
    df_cv.to_csv(os.path.join(output_dir, f"CV_Mean_{name}.csv"))

# --------------------------
# Combine CVs into long format for plotting
# --------------------------
df_long = pd.DataFrame(columns=['Method', 'Group', 'CV (%)'])

for name, df_cv in df_cv_dict.items():
    for group_name in groups.keys():
        df_tmp = pd.DataFrame({
            'Method': name,
            'Group': group_name,
            'CV (%)': df_cv[f'cv_{group_name}']
        })
        df_long = pd.concat([df_long, df_tmp], ignore_index=True)

# --------------------------
# Plot CV distributions side by side
# --------------------------
plt.figure(figsize=(14, 10))
palette = sns.color_palette("Set2", n_colors=len(df_long['Method'].unique()))
sns.stripplot(
    x='CV (%)',
    y='Method',
    hue='Group',
    data=df_long,
    dodge=True,
    size=5,
    jitter=True,
    alpha=0.7,
    palette=palette
)
plt.xlabel('Coefficient of Variation (%)', fontsize=16)
plt.ylabel('')
plt.title('CV Distribution Across T0 and T6 Groups (All Transformations)', fontsize=16)
plt.legend(title='Group')
plt.tight_layout()
plt.show()








