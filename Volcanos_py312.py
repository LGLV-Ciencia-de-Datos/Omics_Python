
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from adjustText import adjust_text
import random


df = pd.read_csv("/Users/Eliel/Documents/Python/Diplomado/Diplomado_2/Block_3/TTEST_example_1.csv")
df.head()

df.columns

# Select 'log2FC' and 'pvals' columns





#Building up volcano plot

plt.rcParams['svg.fonttype'] = 'none'

ax= sns.scatterplot(data = df,  x= 'log2FC_T6_T0', y = 'transformed_pval', label="Not significant", color="lightgrey")
ax.axhline(2.0, zorder = 0, c = 'k', lw = 1, ls ='--')
ax.axvline(0.4, zorder = 0, c = 'k', lw = 1, ls ='--')
ax.axvline(-0.2, zorder = 0, c = 'k', lw = 1, ls ='--')


# highlight down- or up- regulated genes
down = df[(df['log2FC_T6_T0']<=-0.2)&(df['transformed_pval']>=1.5)]
up = df[(df['log2FC_T6_T0']>=0.4)&(df['transformed_pval']>=1.5)]

ax= sns.scatterplot(data = df,  x= down['log2FC_T6_T0'], y = down['transformed_pval'], label="Down-regulated", color="blue")
ax= sns.scatterplot(data = df,  x= up['log2FC_T6_T0'], y = up['transformed_pval'], label="Up-regulated", color="red")


plt.show()




#other color

ax= sns.scatterplot(data = df,  x= 'log2FC_T6_T0', y = 'transformed_pval', label="Not significant", color="lightgrey")
ax.axhline(2.0, zorder = 0, c = 'k', lw = 1, ls ='--')
ax.axvline(0.4, zorder = 0, c = 'k', lw = 1, ls ='--')
ax.axvline(-0.2, zorder = 0, c = 'k', lw = 1, ls ='--')

ax= sns.scatterplot(data = df,  x= down['log2FC_T6_T0'], y = down['transformed_pval'], label="Down-regulated", color="navy")
ax= sns.scatterplot(data = df,  x= up['log2FC_T6_T0'], y = up['transformed_pval'], label="Up-regulated", color="gold")
ax.tick_params(axis='both', labelsize=12)



texts = []
for i in range (len(df)):
    if df.iloc[i].transformed_pval > 2.5 and abs(df.iloc[i].log2FC_T6_T0) > 0.4:
        texts.append(plt.text(x = df.iloc[i].log2FC_T6_T0, y = df.iloc[i].transformed_pval, s = df.iloc[i].ids,
                                  fontsize = 10, weight = 'regular'))
        


adjust_text(texts, # expand text bounding boxes by 1.2 fold in x direction and 2 fold in y direction
            arrowprops=dict(arrowstyle='-', color='k') # ensure the labeling is clear by adding arrows
            )

plt.show()




#--------------------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def volcano_plot(
    df,
    fc_col="log2FC_T6_T0",
    pval_col="transformed_pval",
    gene_col="ids",
    fc_thresh=0.4,
    pval_thresh=1.3,
    label_thresh=2.5,
    figsize=(8, 7)
):
    """
    Reusable volcano plot function
    """

    # Create copy
    data = df.copy()

    # Define significance
    data["significance"] = "Not significant"

    data.loc[
        (data[fc_col] >= fc_thresh) & (data[pval_col] >= pval_thresh),
        "significance"
    ] = "Up"

    data.loc[
        (data[fc_col] <= -fc_thresh) & (data[pval_col] >= pval_thresh),
        "significance"
    ] = "Down"

    # Color map
    colors = {
        "Not significant": "lightgrey",
        "Up": "red",
        "Down": "blue"
    }

    # Plot
    plt.figure(figsize=figsize)

    for group in ["Not significant", "Up", "Down"]:
        subset = data[data["significance"] == group]

        plt.scatter(
            subset[fc_col],
            subset[pval_col],
            c=colors[group],
            label=group,
            alpha=0.7,
            s=30
        )

    # Threshold lines
    plt.axhline(pval_thresh, color="black", linestyle="--")
    plt.axvline(fc_thresh, color="black", linestyle="--")
    plt.axvline(-fc_thresh, color="black", linestyle="--")

    # Labels
    plt.xlabel("Log2 Fold Change")
    plt.ylabel("-Log10 p-value")
    plt.title("Volcano Plot")

    # Annotate top hits
    for _, row in data.iterrows():
        if row[pval_col] > label_thresh and abs(row[fc_col]) > fc_thresh:
            plt.text(
                row[fc_col],
                row[pval_col],
                row[gene_col],
                fontsize=8
            )

    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# Example usage
df = pd.read_csv("/Users/Eliel/Documents/Python/Diplomado/Diplomado_2/Block_3/TTEST_example_2.csv")

volcano_plot(df)


#--------------------------------------
#Interactive Volcano (Zoom + Hover) – With Plotly
#----------------------------------------------


import plotly.io as pio
pio.renderers.default = "browser"




import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"   # <<< ADD THIS

df = pd.read_csv("/Users/Eliel/Documents/Python/Diplomado/Diplomado_2/Block_3/TTEST_example_2.csv")

df["status"] = "Not significant"

df.loc[
    (df["log2FC_T6_T0"] >= 0.4) & (df["transformed_pval"] >= 1.3),
    "status"
] = "Up"

df.loc[
    (df["log2FC_T6_T0"] <= -0.4) & (df["transformed_pval"] >= 1.3),
    "status"
] = "Down"


fig = px.scatter(
    df,
    x="log2FC_T6_T0",
    y="transformed_pval",
    color="status",
    hover_name="ids",
    color_discrete_map={
        "Up": "red",
        "Down": "blue",
        "Not significant": "lightgrey"
    },
    title="Interactive Volcano Plot"
)

fig.add_hline(y=1.3, line_dash="dash")
fig.add_vline(x=0.4, line_dash="dash")
fig.add_vline(x=-0.4, line_dash="dash")

fig.show()




#-----------------------------------------
#--------------Density Volcano (KDE-Based)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/Users/Eliel/Documents/Python/Diplomado/Diplomado_2/Block_3/TTEST_example_2.csv")

plt.figure(figsize=(8,7))

# Density background
sns.kdeplot(
    x=df["log2FC_T6_T0"],
    y=df["transformed_pval"],
    fill=True,
    cmap="viridis",
    levels=100,
    thresh=0.05
)

# Significant UP (positive fold change)
up = df[
    (df["log2FC_T6_T0"] > 0.4) &
    (df["transformed_pval"] > 1.3)
]

# Significant DOWN (negative fold change)
down = df[
    (df["log2FC_T6_T0"] < -0.4) &
    (df["transformed_pval"] > 1.3)
]

# Plot UP (Blue)
plt.scatter(
    up["log2FC_T6_T0"],
    up["transformed_pval"],
    c="blue",
    s=25,
    label="Up-regulated"
)

# Plot DOWN (Red)
plt.scatter(
    down["log2FC_T6_T0"],
    down["transformed_pval"],
    c="red",
    s=25,
    label="Down-regulated"
)

# Thresholds
plt.axhline(1.3, ls="--", c="black")
plt.axvline(0.4, ls="--", c="black")
plt.axvline(-0.4, ls="--", c="black")

plt.xlabel("Log2 Fold Change")
plt.ylabel("-Log10 p-value")
plt.title("Density Volcano Plot (Up/Down Highlighted)")
plt.legend()

plt.tight_layout()
plt.show()











#----------------
#Hexbin Volcano (Best for Massive Data)


import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/Eliel/Documents/Python/Diplomado/Diplomado_2/Block_3/TTEST_example_2.csv")

plt.figure(figsize=(8,7))

hb = plt.hexbin(
    df["log2FC_T6_T0"],
    df["transformed_pval"],
    gridsize=60,
    cmap="inferno",
    mincnt=1
)

plt.colorbar(hb, label="Counts")

# Thresholds
plt.axhline(1.3, ls="--", c="white")
plt.axvline(0.4, ls="--", c="white")
plt.axvline(-0.4, ls="--", c="white")

plt.xlabel("Log2 Fold Change")
plt.ylabel("-Log10 p-value")
plt.title("Hexbin Volcano Plot")

plt.show()


#------------------
#MA–Volcano Hybrid (Expression + Significance)
#“How big is the change vs how abundant is the protein?”
#upload_adundances


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/Eliel/Documents/Python/Diplomado/Diplomado_2/Block_3/CV_Mean_RobustScaler.csv")

# Mean expression
df["mean_expr"] = (df["mean_T0"] + df["mean_T6"]) / 2 #we want one number per protein that represents:Its overall abundance level

colors = []

for i in range(len(df)):

    fc = df.loc[i, "log2FC_T6_T0"]
    pval = df.loc[i, "transformed_pval"]

    # Significant up
    if fc > 0.4 and pval > 1.3:
        colors.append("blue")

    # Significant down
    elif fc < -0.4 and pval > 1.3:
        colors.append("red")

    # Not significant
    else:
        colors.append("lightgrey")


plt.figure(figsize=(8,7))

plt.scatter(
    df["mean_expr"],
    df["log2FC_T6_T0"],
    c=colors,
    alpha=0.7,
    s=40
)

# Thresholds
plt.axhline(0.4, ls="--", c="black")
plt.axhline(-0.4, ls="--", c="black")

plt.xlabel("Mean Expression (T0 & T6)")
plt.ylabel("Log2 Fold Change (T6 vs T0)")
plt.title("MA–Volcano Hybrid Plot")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()











