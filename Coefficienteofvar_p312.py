
import pandas as pd
import numpy as np
import re 
import plotly.graph_objects as go
import math
import scipy.stats as stat
import pandas as pd


df= pd.read_csv('Box-Cox_knn_Data1.csv')
df

df.shape

X = df.iloc[:, 1:7]# del index a la ultima fila

X


X.min()


#add 1.6 to all data frame


df2= X + 4
df2



#use this when no negative values ######
df2=X
###########



#########


column_names = list(df2.columns.values)
column_names
column_names.pop(0) 


groups = {}

index = 1
for column_name in column_names:
    group_name = re.sub('_[0-9]$', '', column_name)#regular expressions are small stringlike expressions that you can use to specify patterns that you want to extract from a string. In our case the pattern we want is any characters followed by _ followed by one or more numeric digits until the end. The regular expression for this is "_[0-9]$"

    if group_name in groups:
        groups[group_name].append(index)
    else:
        groups[group_name] = [index];
    
    index = index + 1

print(groups)




groupnames = list(groups.keys())

groupnames

g1_col_indices = groups[groupnames[0]]#DMSO
g1_col_indices

g2_col_indices = groups[groupnames[1]]#ARID1A
g2_col_indices





g1_values = df2.iloc[:, g1_col_indices].to_numpy()
g1_values

g2_values = df2.iloc[:, g2_col_indices].to_numpy()
g2_values




cv_list_1 = []


for i in range(g1_values.shape[0]): # Iterate through each row index
    row_data = g1_values[i, :] # Get data for the current row
    
    # Calculate standard deviation and mean for the current row
    std_dev = np.std(row_data)
    mean = np.mean(row_data)
    
    # Calculate CV, handling potential division by zero
    if mean != 0:
        cv = (std_dev / mean) * 100
    else:
        cv = np.nan # Or some other appropriate value for undefined CV
    
    cv_list_1.append(cv)


cv_list_1

#==================
cv_list_2 = []
for i in range(g2_values.shape[0]): # Iterate through each row index
    row_data = g2_values[i, :] # Get data for the current row
    
    # Calculate standard deviation and mean for the current row
    std_dev = np.std(row_data)
    mean = np.mean(row_data)
    
    # Calculate CV, handling potential division by zero
    if mean != 0:
        cv = (std_dev / mean) * 100
    else:
        cv = np.nan # Or some other appropriate value for undefined CV
    
    cv_list_2.append(cv)


cv_list_2




#list to be added as a new column
List1 = cv_list_1
List2 = cv_list_2

# Add the list as a new column 
df2['cv1'] = List1
df2['cv2'] = List2




g1_means = g1_values.mean(axis = 1).tolist()
g1_means

g2_means = g2_values.mean(axis = 1).tolist()
g2_means


# Add means
df2['means1'] = g1_means
df2['means2'] = g2_means


df2.to_csv('Box-Cox_cv.csv', index=0)

