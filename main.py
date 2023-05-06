import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

# read data file
df = pd.read_csv('data.csv')

# round each column to 3 decimals
df = df.round(3)

# get dataframe lines without headers
print(df.shape)

# get the number of null values in each column
print(df.isnull().sum())

print(df.describe().round(2).to_string())
print(df.select_dtypes(include='number').mean())

mean_values = df.groupby('Entity')[['Daily tests', 'Cases', 'Deaths']].mean()
print(mean_values)

var_values = df.groupby('Entity')[['Daily tests', 'Cases', 'Deaths']].var()
print(var_values)

# non_numeric = df['Daily tests'].select_dtypes(exclude='number')
# print(non_numeric)
# for column in df.drop(['Entity', 'Continent', 'Date', 'Cases', 'Deaths'], axis=1):
#    plt.figure()
#    df.boxplot([column])
#    plt.show()

columns = ['Date', 'Daily tests', 'Cases', 'Deaths']
df.drop(columns, axis=1).drop_duplicates().hist(bins=15, figsize=(16, 9), rwidth=0.8)
plt.show()

# Keep the last line (date) for each country and drop unused columns
df_last = df.groupby('Entity').tail(1).drop(['Entity', 'Date'], axis=1)
# plt.figure(figsize=(12, 8))
sns.heatmap(df_last.corr(numeric_only=True), annot=True, cmap=plt.cm.Reds)
plt.show()

