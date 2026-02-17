import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Data1.csv")
df

df.shape


#boxplot
#muestra la distribución de datos numéricos usando sus cuartiles, 
#indicando la mediana, el rango intercuartil (IQR) y posibles valores atípicos.

#La caja representa el 50% central de los datos, con una línea que marca la mediana

#Los bigotes se extienden hasta los valores máximo y mínimo,

#los valores atípicos, que se representan como puntos fuera de los bigotes

ax = sns.boxplot( data = df.iloc[:, 0:7])
ax.set_ylabel('Example', fontsize = 16)
plt.xticks(rotation=45, fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()



#violin plot
#La forma del "violín" representa la distribución de la densidad de los datos, 
#siendo más ancha donde hay más datos y más estrecha donde hay menos
#el diagrama de violín también muestra la forma de la distribución en detalle. 

ax = sns.violinplot( data = df.iloc[:, 0:7])
ax.set_ylabel('Example', fontsize = 16)
plt.xticks(rotation=45, fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()


#stripplot
#es un gráfico de dispersión donde los puntos se "agitan" para evitar la superposición, 
#permitiendo ver la distribución individual de cada categoría. 


ax = sns.stripplot( data = df.iloc[:, 0:7])
ax.set_ylabel('Example', fontsize = 16)
plt.xticks(rotation=45, fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()

#stripplot h en orientacion para obtener fraficos horizontales

ax = sns.stripplot( data = df.iloc[:, 0:7], orient='h')
ax.set_ylabel('Example', fontsize = 16)
plt.xticks(rotation=45, fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()



#swarmplot. Considerar datos no tan abundantes
#Visualización de la distribución. Este ajuste permite ver la distribución de los datos dentro de cada categoría de manera más clara,
# mostrando dónde se concentran los puntos y si hay valores atípicos. 

df = pd.read_csv("Data2.csv")
df


ax = sns.swarmplot( data = df.iloc[:, 0:7])
ax.set_ylabel('Example', fontsize = 16)
plt.xticks(rotation=45, fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()




#======================================================================

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

