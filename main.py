import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# read data file
df = pd.read_csv('data.csv')

# get the non common dates and remove them from the dataframe
df['Date'] = pd.to_datetime(df['Date'])
dates = df.groupby('Date').size()
unique_countries = df['Entity'].unique()

outdates = []
for i in range(0, len(dates)):
    if dates.values[i] < len(unique_countries):
        outdates.append(dates.index[i])

print(outdates)
mask = df['Date'].isin(outdates)
df_filtered = df[~mask]

print(df_filtered.info())

# get the number of null values in each column
print("Number of null values in each column: ")
print(df_filtered.isnull().sum())

# Describe values for each column: number of values, min, max, mean, standard deviation, Q1, Q2 and Q3 quartiles
print("Describe: \n", df_filtered.describe().round(3).to_string())

# We set the criteria for a country to have poor data, we find those countries and remove them from the dataframe
rows_per_country = df_filtered.groupby('Entity').size()
print(rows_per_country.values)

nulls_per_country = df_filtered.groupby('Entity').apply(lambda x: x.isnull().sum())
print(nulls_per_country)

# store and print the countries where missing data from columns 'Daily tests','Cases','Deaths' are more than half country's data
bad_data = 0
countries_to_remove = []
for i in range(0, 104):
    for j in range(12, 15):
        # print("->", rows_per_country.values[i] - nulls_per_country.values[i][j])
        if (rows_per_country.values[i] - nulls_per_country.values[i][j]) < rows_per_country.values[i]/2:
            bad_data = 1
    if bad_data:
        countries_to_remove.append(nulls_per_country.index[i])
        print("Country name: ", nulls_per_country.index[i], ",Total values: ", rows_per_country.values[i],
              ",Nulls: Daily tests: ", nulls_per_country.values[i][12], ",Cases: ", nulls_per_country.values[i][13],
              ",Deaths: ", nulls_per_country.values[i][14])
        bad_data = 0

# remove from dataframe all countries found in the previous step
for country_name in countries_to_remove:
    df_filtered = df_filtered.drop(df_filtered.loc[df['Entity'] == country_name].index)

print("Number of null values in each column: ")
print(df_filtered.isnull().sum())

# df_filtered.to_csv('clean_data.csv', index=False)

nulls_per_country = df_filtered.groupby('Entity').apply(lambda x: x.isnull().sum())
print(nulls_per_country)

groups = df_filtered.groupby('Entity')

df_filtered[['Daily tests', 'Cases', 'Deaths']] = df_filtered[['Daily tests', 'Cases', 'Deaths']].interpolate(method='linear', limit_direction='forward')
df_filtered[['Daily tests', 'Cases', 'Deaths']] = df_filtered[['Daily tests', 'Cases', 'Deaths']].round()

# df_filtered['Cases'] = df_filtered['Cases'].interpolate(method='')
df_filtered.to_csv('interpolated_data.csv', index=False)

print("Number of null values in each column: ")
print(df_filtered.isnull().sum())

