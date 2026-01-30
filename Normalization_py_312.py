

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 10:04:55 2025

@author: Eliel
"""

#Installing scikit-learn in anaconda terminal







import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#reading database

pd.set_option('display.expand_frame_repr', False)


df= pd.read_csv('imputeed-knn_Data1.csv',  index_col=0)
df



#Standarization

from sklearn.preprocessing import StandardScaler

scaleStandard = StandardScaler()


X1 = scaleStandard.fit_transform(df)
X1


#Creating a dataframe

columns= df.columns
columns


Standard = pd.DataFrame(X1, columns= columns, index=df.index) 
Standard



Standard.to_csv('standard_knn_Data1.csv')



#normalization Minimax-SCALING TO A RANGE


from sklearn.preprocessing import MinMaxScaler

ScaleMinMax = MinMaxScaler(feature_range = (1,2))


X2 = ScaleMinMax.fit_transform(df)
X2


#Creating a dataframe


Minimax = pd.DataFrame(X2, columns= columns, index=df.index) 

Minimax.to_csv('Minimax_knn_Data1.csv')



## Apply QuantileTransformer-SCALING TO A DISTRIBUTION


from sklearn.preprocessing import QuantileTransformer

ScaleQuantile = QuantileTransformer (n_quantiles=465, output_distribution='normal', random_state=0)

X3 = ScaleQuantile.fit_transform(df)
X3

#Creating a dataframe


Quantile = pd.DataFrame(X3, columns= columns, index=df.index) 

Quantile.to_csv('Quantile_knn_Data1.csv')


# RobustScaler-SCALING TO A DISTRIBUTION

from sklearn import preprocessing


scaler = preprocessing.RobustScaler()

X4 = scaler.fit_transform(df)
X4

#Creating a dataframe


Robust = pd.DataFrame(X4, columns= columns, index=df.index)
Robust


Robust.to_csv('Robust_knn_ata1.csv')


# Power Transformations-Yeo-Johnson (for positive or negative data) -PARAMETRIC SCALING
#Variance Stabilizing Normalization (VSN) in Python aims to transform data so that its variance becomes independent of its mean, 
# which is particularly useful for count data or data with heteroscedasticity. Several methods and libraries in Python can be used to achieve this:


from sklearn.preprocessing import PowerTransformer



# Apply Yeo-Johnson transformation
pt = PowerTransformer(method='yeo-johnson')

transformed_data = pt.fit_transform(df)
transformed_data

PowerTrans = pd.DataFrame(transformed_data, columns= columns, index=df.index)

PowerTrans.to_csv('Y-J_knn_Data1.csv')



# Power Transformations-Box-Cox (for strictly positive data)
# 
# from sklearn.preprocessing import PowerTransformer



# Apply Yeo-Johnson transformation

pt2 = PowerTransformer(method='box-cox')
transformed_data2 = pt.fit_transform(df)
transformed_data2


transformed_data2 = pd.DataFrame(transformed_data2, columns= columns, index=df.index)


transformed_data2.to_csv('Box-Cox_knn_Data1.csv')


#square root transformation-SCALING TO A SHAPE


import numpy as np

transformed_data_sqrt = np.sqrt(df)
transformed_data_sqrt

sqrt = pd.DataFrame(transformed_data_sqrt, columns= columns, index=df.index)


sqrt.to_csv('SQRT_knn_Data1.csv')



###Plotting 

plt.rcParams['svg.fonttype'] = 'none'

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 19))
plt.subplots_adjust(hspace=0.5)


sns.violinplot(ax=axes[0, 0], data = df.iloc[:, :6])#number of columns
axes[0, 0].set_title('Original Data')
axes[0, 0].tick_params(axis='x', labelrotation=60)


sns.violinplot(ax=axes[1, 0], data = Standard.iloc[:, :6])
axes[1, 0].set_title('StandardScaler')
axes[1, 0].tick_params(axis='x', labelrotation=60)

sns.violinplot(ax=axes[2, 0], data = Minimax.iloc[:, :6])
axes[2, 0].set_title('MinMaxScaler')
axes[2, 0].tick_params(axis='x', labelrotation=60)


sns.violinplot(ax=axes[3, 0], data = Quantile.iloc[:, :6])
axes[3, 0].set_title('QuantileTransformer')
axes[3, 0].tick_params(axis='x', labelrotation=60)


sns.violinplot(ax=axes[0, 1], data = Robust.iloc[:, :6])
axes[0, 1].set_title('RobustScaler')
axes[0, 1].tick_params(axis='x', labelrotation=60)

sns.violinplot(ax=axes[1, 1], data = PowerTrans.iloc[:, :6])
axes[1, 1].set_title('PowerTrans-yeo-johnson')
axes[1, 1].tick_params(axis='x', labelrotation=60)

sns.violinplot(ax=axes[2, 1], data = transformed_data2.iloc[:, :6])
axes[2, 1].set_title('PowerTrans-Box-Cox')
axes[2, 1].tick_params(axis='x', labelrotation=60)

sns.violinplot(ax=axes[3, 1], data = sqrt.iloc[:, :6])
axes[3, 1].set_title('SQTR')
axes[3, 1].tick_params(axis='x', labelrotation=60)



plt.show()
