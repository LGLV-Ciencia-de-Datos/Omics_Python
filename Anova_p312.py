
import pandas as pd
import numpy as np
import re 
import plotly.graph_objects as go
import math
import scipy.stats as stat
from scipy.stats import f_oneway


seq_df1= pd.read_csv('/Users/Eliel/Documents/Python/Diplomado/Bloque_3/morethan2groips/RF/cv/Positive-Quantile.csv')

seq_df1


#let’s remove the first value of the list which will contain the name of the first column "Identifiers" which we do not want since it is not one of our conditions.

column_names = list(seq_df1.columns.values)
column_names
column_names.pop(0) 


#Grouping columns/data based on conditions

#Now lets construct a loop to go through each of our columns/condition names and create groups that contain all samples associated with a specific condition. 

groups = {}

index = 1
for column_name in column_names:
    group_name = re.sub('_[0-9]$', '', column_name)#egular expressions are small stringlike expressions that you can use to specify patterns that you want to extract from a string. In our case the pattern we want is any characters followed by _ followed by one or more numeric digits until the end. The regular expression for this is "_[0-9]$"

    if group_name in groups:
        groups[group_name].append(index)
    else:
        groups[group_name] = [index];
    
    index = index + 1

print(groups)

#Calculating the fold changes    


#Assume for now we just want to compare DMSO with ARID1A,o to get that data we need to 
#grab the columns that correspond to each group. An easy way to do that is to use the .keys() function to 
#get the keys (condition names) from our groups dictionary and convert it to a list. 
#The resulting list will have the first group name at index 0, the second at index 1, etc.

groupnames = list(groups.keys())
groupnames


g1_col_indices = groups[groupnames[0]]#DMSO
g1_col_indices

g2_col_indices = groups[groupnames[1]]#ARID1A
g2_col_indices

g3_col_indices = groups[groupnames[2]]#ARID1A
g3_col_indices

g4_col_indices = groups[groupnames[3]]#ARID1A
g4_col_indices

g5_col_indices = groups[groupnames[4]]#ARID1A
g5_col_indices


#We will turn again to the pandas .iloc function and just like before we will get 
#all dataframes rows by passing in the colon and we will pass in the the column indices to
# get only the columns corresponding to the data we want.


g1_values = seq_df1.iloc[:, g1_col_indices].to_numpy()
g2_values = seq_df1.iloc[:, g2_col_indices].to_numpy()
g3_values = seq_df1.iloc[:, g3_col_indices].to_numpy()
g4_values = seq_df1.iloc[:, g4_col_indices].to_numpy()
g5_values = seq_df1.iloc[:, g5_col_indices].to_numpy()

#To get the mean of each row (which corresponds to the mean expression value of 
#a gene in a condition) we can use the numpy mean function and specify axis = 1. 
#This tells numpy to calculate the mean of each of the rows

g1_means = g1_values.mean(axis = 1)
g2_means = g2_values.mean(axis = 1)
g3_means = g3_values.mean(axis = 1)
g4_means = g4_values.mean(axis = 1)
g5_means = g5_values.mean(axis = 1)


#With the means in hand, we're ready to calculate the log2 of the fold changes between group 1 and group 2

foldchanges1= list(np.log2(np.divide(g2_means, g1_means))) #(g2/g1)
foldchanges1

foldchanges2= list(np.log2(np.divide(g3_means, g2_means))) #(g3/g2)
foldchanges2

foldchanges3= list(np.log2(np.divide(g4_means, g3_means))) #(g4/g3)
foldchanges3

foldchanges4= list(np.log2(np.divide(g5_means, g4_means))) #(g5/g4)
foldchanges4


#Calculating the p-values anova

pvals = []
num_values = g1_means.shape[0]
for row in range(0, num_values):
    oneway_result = f_oneway(g1_values[row, :],  g2_values[row, :], g3_values[row, :], g4_values[row, :], g5_values[row, :]) # check how anova one way works, Kruskal-Wallis test,  fo no normal distribution
    pvalue = oneway_result [1]
    pvals.append(pvalue)
    
pvals


# transforms pvalues with -log10

transformed_pvals = list(-1*np.log10(pvals))
transformed_pvals


# Bonferroni correction

# Calculate Bonferroni-adjusted p-values


adjusted_p_values_BC = np.array(pvals)* len(pvals)
adjusted_p_values_BC

adjusted_p_values_BC2=list(-1*np.log10 (adjusted_p_values_BC))
adjusted_p_values_BC2




# Apply Benjamini-Hochberg (BH) FDR correction

from scipy import stats


adjusted_p_values_bh = stats.false_discovery_control((pvals), method='bh')

adjusted_p_values_bh2= list(-1*np.log10 (adjusted_p_values_bh))
adjusted_p_values_bh2


# Apply Benjamini-Yekutieli (BY) FDR correction (more conservative)
adjusted_p_values_by = stats.false_discovery_control(pvals, method='by')
adjusted_p_values_by2=list(-1*np.log10 (adjusted_p_values_by))
adjusted_p_values_by2




arr1 = np.array(foldchanges1)

arr2 = np.array(foldchanges2)

arr3 = np.array(foldchanges3)

arr4 = np.array(foldchanges4)

arr5 = np.array(transformed_pvals)

arr6 = np.array(adjusted_p_values_BC2)



arr7 = np.array(adjusted_p_values_bh2)

arr8 = np.array(adjusted_p_values_by2)


arrX = np.stack((arr1, arr2, arr3, arr4, arr5, arr6), axis=1)
arrX


#Convert array in a dataframe 

df = pd.read_csv('/Users/Eliel/Documents/Python/Diplomado/Bloque_3/morethan2groips/RF/cv/Positive-Quantile.csv', index_col=0)

columns = ['log2FC 2cm/1cm','log2FC 3cm/2cm','log2FC 4cm/3cm','log2FC 5cm/4cm',
           'transformed_pval', 'adjusted_p_values_BC2']

X1 = pd.DataFrame(arrX, columns = columns, index=df.index)

X1

                           
X1.to_csv('Anova_example.csv')




