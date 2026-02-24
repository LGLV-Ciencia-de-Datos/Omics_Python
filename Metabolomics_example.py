#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 20:37:54 2026

@author: Eliel
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from collections import defaultdict
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer

# ========= 1. Load Data =========
df = pd.read_csv("Data5.csv", index_col=0)
output_dir = "results_auto"
os.makedirs(output_dir, exist_ok=True)

# ========= 2. Intensity columns & groups (auto-detect) =========
intensity_cols = [col for col in df.columns if "_" in col and col != "Feature_ID"]
X = df[intensity_cols].replace(0, np.nan)

groups = defaultdict(list)
for col in intensity_cols:
    prefix = col.split('_')[0]
    groups[prefix].append(col)
groups = dict(groups)
print("Auto-detected groups (prefixes):", groups)

# ========= 3. Impute (KNN) =========
imputer = KNNImputer(n_neighbors=3)
X_knn = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
X_knn.to_csv(os.path.join(output_dir, "imputed_KNN.csv"))
print("Saved imputed_KNN.csv")

# ========= 4. Universal transformation wrapper =========
def apply_positive_transform(name, transformer, data, columns, save=True, outdir="results_auto"):
    transformed = transformer.fit_transform(data)
    result = pd.DataFrame(transformed, columns=columns, index=data.index)
    min_val = result.min().min()
    if min_val <= 0:
        shift = abs(min_val) + 1e-3
        result += shift
        print(f"[{name}] Shift applied: {shift}")
    if save:
        result.to_csv(os.path.join(outdir, f"{name}_intensities.csv"))
    return result

# ========= 5. Apply transformations and keep strictly positive =========
transformed_dfs = {}
# Also add the raw KNN imputed data (ensure minimum is positive)
min_knn = X_knn.min().min()
if min_knn <= 0:
    shift = abs(min_knn) + 1e-3
    X_knn_pos = X_knn + shift
    print(f"[Imputed_KNN] Shift applied: {shift}")
else:
    X_knn_pos = X_knn.copy()
transformed_dfs["Imputed_KNN"] = X_knn_pos

transformed_dfs["Standard"] = apply_positive_transform("Standard", StandardScaler(), X_knn, X_knn.columns)
transformed_dfs["Robust"] = apply_positive_transform("Robust", RobustScaler(), X_knn, X_knn.columns)
transformed_dfs["BoxCox"] = apply_positive_transform("BoxCox", PowerTransformer(method="box-cox"), X_knn, X_knn.columns)

# ========= 6. Compute mean & CV for all groups/scalers (means are strictly positive) =========
def compute_group_cv(df, groups):
    df_cv = df.copy()
    for group_name, cols in groups.items():
        vals = df[cols].to_numpy()
        means = vals.mean(axis=1)
        stds = vals.std(axis=1)
        cv = np.where(means > 0, (stds / means) * 100, np.nan)  # means always positive after transform!
        df_cv[f"mean_{group_name}"] = means
        df_cv[f"cv_{group_name}"] = cv
    return df_cv

df_cv_dict = {}
print("\nComputing CVs for all scalers and groups...")
for name, df_trans in transformed_dfs.items():
    df_cv = compute_group_cv(df_trans, groups)
    df_cv_dict[name] = df_cv
    out_path = os.path.join(output_dir, f"CV_Mean_{name}.csv")
    df_cv.to_csv(out_path)
    print(f"Saved: {out_path}")

# ========= 7. Convert CVs to long format for plotting, filter only positive CVs =========
df_long = pd.DataFrame(columns=["Method", "Group", "CV (%)"])
for name, df_cv in df_cv_dict.items():
    for group_name in groups.keys():
        df_tmp = pd.DataFrame({
            "Method": name,
            "Group": group_name,
            "CV (%)": df_cv[f"cv_{group_name}"]
        })
        df_long = pd.concat([df_long, df_tmp], ignore_index=True)

df_long_pos = df_long[df_long["CV (%)"] > 0]

# ========= 8. Plot all positive CVs per group and scaler (including Imputed_KNN) =========
plt.figure(figsize=(14, 9))
sns.stripplot(
    x="CV (%)",
    y="Method",
    hue="Group",
    data=df_long_pos,
    dodge=True,
    size=4,
    jitter=0.15,
    alpha=0.7
)
plt.xlabel("Coefficient of Variation (%)", fontsize=14)
plt.ylabel("Scaling Method", fontsize=13)
plt.title(
    "CV Distribution Across Groups (Imputed_KNN, Standard, Robust, Box-Cox)\nOnly Positive CV",
    fontsize=15
)
plt.legend(title="Group", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()






#------------------------
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import normaltest, anderson, probplot

# --- Function for Normality Test and Plotting ---
def test_and_plot_normality(dataset, col_name, alpha=0.05, anderson_significance_idx=2):
    data = dataset[col_name].dropna()
    n = len(data)
    print(f"\nColumn: {col_name} (N = {n})")
    if n < 8:
        print("Warning: Too few samples for D'Agostino-Pearson normality test (< 8).")
        return

    # D'Agostino-Pearson
    stat, p_value = normaltest(data)
    print(f"D'Agostino-Pearson Test Statistic: {stat:.3f}")
    print(f"P-value: {p_value:.3f}")

    if p_value > alpha:
        dp_decision = "Normal"
    else:
        dp_decision = "Not Normal"

    print(f"Decision at alpha={alpha}: {dp_decision}")

    # Anderson-Darling
    result = anderson(data)
    ad_stat = result.statistic
    ad_crit = result.critical_values[anderson_significance_idx]
    ad_level = result.significance_level[anderson_significance_idx]
    print(f"\nAnderson-Darling Test Statistic: {ad_stat:.3f}")
    print(f"Critical Value at {ad_level}%: {ad_crit:.3f}")
    ad_decision = "Normal" if ad_stat < ad_crit else "Not Normal"
    print(f"Decision at {ad_level}%: {ad_decision}")

    # --- Plot histogram with annotation + Q-Q plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(data, bins=40, kde=True, color='blue', alpha=0.6, ax=axes[0])
    axes[0].set_title(f"Distribution of {col_name}", fontsize=15)
    axes[0].set_xlabel(col_name, fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].text(0.98, 0.98, f"D'Agostino-Pearson P={p_value:.3g}\nDecision: {dp_decision}", 
                 transform=axes[0].transAxes, ha='right', va='top', 
                 bbox=dict(boxstyle="round,pad=0.2",facecolor="white", alpha=0.8))

    axes[0].text(0.98, 0.85, f"Anderson-Darling Stat={ad_stat:.2f}\nDecision: {ad_decision}", 
                 transform=axes[0].transAxes, ha='right', va='top', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.2",facecolor="white", alpha=0.85))
    
    # Q-Q plot for visual normality
    probplot(data, dist="norm", plot=axes[1])
    axes[1].set_title(f"Q-Q Plot of {col_name}", fontsize=15)
    plt.tight_layout()
    plt.show()

# --- Usage Example ---
dataset = pd.read_csv('/Users/Eliel/Documents/Python/Diplomado/Diplomado_2/Block_3/Metabomics_example/results_auto/imputed_KNN.csv', index_col=0)
dataset.columns
# Change '1cm_1' to any column you want to check!
test_and_plot_normality(dataset, col_name='1cm_1')




#-----Differetial analysis 



import pandas as pd
import numpy as np
import re
from collections import defaultdict
from scipy.stats import kruskal
import statsmodels.stats.multitest as smm

# Load data
seq_df1 = pd.read_csv('/Users/Eliel/Documents/Python/Diplomado/Diplomado_2/Block_3/Metabomics_example/results_auto/imputed_KNN.csv', index_col=0)

# Auto-detect columns, group by prefix before "_"
intensity_cols = [col for col in seq_df1.columns if "_" in col and col.lower() not in ['feature_id', 'identifiers', 'id']]
groups = defaultdict(list)
for col in intensity_cols:
    prefix = col.split('_')[0]
    groups[prefix].append(col)
groups = dict(groups)
print("Auto-detected groups (prefixes):", groups)

groupnames = list(groups.keys())
n_groups = len(groupnames)
group_values = [seq_df1.loc[:, groups[gn]].to_numpy() for gn in groupnames]
group_means = [gvals.mean(axis=1) for gvals in group_values]

# Flexible log2FC computation (manual modulation)
def safe_log2fc(numerator, denominator):
    ratio = np.divide(numerator, denominator, where=(denominator > 0), out=np.full_like(numerator, np.nan))
    return np.log2(ratio)

group_lookup = {name: idx for idx, name in enumerate(groupnames)}

group_lookup

# Specify any pairs you want:
fc_pairs_names = [('Large', 'Lung'), ('Giant', 'Large'), ('Small', 'Giant')]

foldchanges = []
for gA, gB in fc_pairs_names:
    idxA = group_lookup[gA]
    idxB = group_lookup[gB]
    fc = safe_log2fc(group_means[idxA], group_means[idxB])
    foldchanges.append(fc)

# Kruskal-Wallis test  
pvals = []
num_features = seq_df1.shape[0]
for i in range(num_features):
    values = [gvals[i, :] for gvals in group_values]
    kruskal_result = kruskal(*values)
    pvals.append(kruskal_result.pvalue)
pvals = np.array(pvals)

# -log10 transform and corrections
transformed_pvals = -np.log10(pvals)
adjusted_p_values_BC = pvals * num_features
adjusted_p_values_BC = np.minimum(adjusted_p_values_BC, 1.0)
adjusted_p_values_BC2 = -np.log10(adjusted_p_values_BC)
bh = smm.multipletests(pvals, method='fdr_bh')[1]
by = smm.multipletests(pvals, method='fdr_by')[1]
adjusted_p_values_bh2 = -np.log10(bh)
adjusted_p_values_by2 = -np.log10(by)

# Combine all and save
arrs = foldchanges + [transformed_pvals, adjusted_p_values_BC2, adjusted_p_values_bh2, adjusted_p_values_by2]
arrX = np.stack(arrs, axis=1)

columns = [f'log2FC {gA}/{gB}' for gA, gB in fc_pairs_names] + [
    'transformed_pval',
    'adjusted_p_values_BC2',
    'adjusted_p_values_bh2',
    'adjusted_p_values_by2'
]

X1 = pd.DataFrame(arrX, columns=columns, index=seq_df1.index)
X1.to_csv('Anova_example.csv')

print("Results saved to Anova_example.csv")



#-------------------


#Pearson cross correlation

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Anova_example.csv',index_col=0)
df

df2=df.corr()


plt.rcParams['svg.fonttype'] = 'none'

cbar_kws = {"shrink":.6,'extend':'both'}

plot= sns.heatmap(df.corr(), cmap="inferno", alpha=0.9, annot=True, cbar_kws=cbar_kws, annot_kws={'size': 10})
plt.show()




#============================================================

#Scatter plots matrix 
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from scipy.stats import pearsonr

# Example: Loading a built-in Seaborn dataset

df = pd.read_csv("Anova_example.csv",index_col=0)
df



g = sns.pairplot(df,kind='reg', diag_kind='kde')
plt.show()



g = sns.pairplot(df, kind='reg', diag_kind='hist')
plt.show()




# Define a function to display correlation

g = sns.pairplot(df,kind='reg', diag_kind='kde')

def corrfunc(x, y, **kwargs):
    r, _ = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate(f"r = {r:.2f}", xy=(.1, .9), xycoords=ax.transAxes, fontsize=10)

# Map the function to the lower triangle of the pairplot
g.map_lower(corrfunc)

plt.show()










