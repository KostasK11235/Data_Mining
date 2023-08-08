import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

# read data file
df = pd.read_csv('data.csv')

# print info about the data
print(df.info())
print(df.describe())

# Plot the distributions of the features in a histogram showing the frequency of values within different ranges (bins)
df.hist(bins=30, figsize=(12, 12))
plt.tight_layout()
plt.show()

# Plot a heatmap for the columns of the dataframe
# We drop columns Entity,Continent and date because they are not numeric
heatmap_df = df.drop(columns=['Entity', 'Continent', 'Date'])
corr_matrix = heatmap_df.corr().round(5)

plt.figure()
sns.heatmap(corr_matrix, annot=True)
plt.title("Dataframe Heatmap")
plt.tight_layout()
plt.show()

# Plot the box-plots for the dataframe
# Calculate total_tests, total_cases, and total_deaths for each country
# Then box-plot the total tests, cases and deaths for all the countries to see the distribution of them
# and find the countries with outlier data
countries_groups = df.groupby("Entity").agg({
    "Daily tests": "sum",
    "Cases": lambda x: x.dropna().iloc[-1],
    "Deaths": lambda x: x.dropna().iloc[-1]
}).reset_index()

print("Countries groups: ", countries_groups)

# Create a boxplot to showcase the distribution of the data
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))

# Boxplot for Total Tests
tests = axes[0].boxplot(countries_groups["Daily tests"])
axes[0].set_yscale('log')
axes[0].set_ylabel('Total Tests')
axes[0].set_title('Total Tests Boxplot Diagram')
axes[0].set_xticks([])

# Boxplot for Total Cases
cases = axes[1].boxplot(countries_groups["Cases"])
axes[1].set_yscale('log')
axes[1].set_ylabel('Total Cases')
axes[1].set_title('Total Cases Boxplot Diagram')
axes[1].set_xticks([])

# Boxplot for Total Deaths
deaths = axes[2].boxplot(countries_groups["Deaths"])
axes[2].set_yscale('log')
axes[2].set_ylabel('Total Deaths')
axes[2].set_title('Total Deaths Boxplot Diagram')
axes[2].set_xticks([])
plt.tight_layout()
plt.show()

# Find and display the outliers(fliers)
tests_outliers = tests["fliers"][0].get_ydata()
cases_outliers = cases["fliers"][0].get_ydata()
deaths_outliers = deaths["fliers"][0].get_ydata()

outliers_countries_tests = countries_groups[countries_groups["Daily tests"].isin(tests_outliers)]["Entity"].tolist()
outliers_countries_cases = countries_groups[countries_groups["Cases"].isin(cases_outliers)]["Entity"].tolist()
outliers_countries_deaths = countries_groups[countries_groups["Deaths"].isin(deaths_outliers)]["Entity"].tolist()

print("Tests Outliers Countries:", outliers_countries_tests)
print("Cases Outliers Countries:", outliers_countries_cases)
print("Deaths Outliers Countries:", outliers_countries_deaths)

# See if there are non common dates among the countries
df['Date'] = pd.to_datetime(df['Date'])
date_groups = df.groupby('Date').size()
unique_countries = df['Entity'].unique()

print("Number of unique countries: ", unique_countries.size)
print("Number of countries in each date group:\n", date_groups)

# Plot the number of times each date appears in the dataset
plt.figure()
date_groups.plot(kind='line')
plt.xlabel('Ordered Dates')
plt.ylabel('Number of appearances')
plt.tight_layout()
plt.show()

# *******************************************************************************************************************
# Preprocessing

# Removing the countries with insufficient data
# Find the number of rows in each country and then the number of null values in each country's columns
rows_per_country = df.groupby('Entity').size()
nulls_per_country = df.groupby('Entity').apply(lambda x: x.isnull().sum())

# store and print the countries where missing data from columns 'Daily tests','Cases','Deaths' are more than half country's data
remove = False
countries_to_remove = []
for i in range(0, 104):
    for j in range(12, 15):
        if (rows_per_country.values[i] - nulls_per_country.values[i][j]) < rows_per_country.values[i]/2:
            remove = True
    if remove:
        countries_to_remove.append(nulls_per_country.index[i])
        remove = False

# remove from dataframe all countries found in the previous step
for country_name in countries_to_remove:
    df = df.drop(df.loc[df['Entity'] == country_name].index)

print("Removed Countries: ", countries_to_remove)

# Get the number of null values in each column (daily tests, cases, deaths) of the cleared dataset
nulls_in_df = df[['Daily tests', 'Cases', 'Deaths']].isnull().sum().values
print("#Daily tests null values: ", nulls_in_df[0])
print("#Cases null values: ", nulls_in_df[1])
print("#Deaths null values: ", nulls_in_df[2])

# Removing all non common dates within the remaining countries
# get the non common dates and remove them from the dataframe

df['Date'] = pd.to_datetime(df['Date'])
dates = df.groupby('Date').size()
unique_countries_number = df['Entity'].nunique()

outdates = []
for i in range(0, len(dates)):
    if dates.values[i] < unique_countries_number:
        outdates.append(dates.index[i])

mask = df['Date'].isin(outdates)
df = df[~mask]

# Print the number of null values in each column
nulls_in_df = df[['Daily tests', 'Cases', 'Deaths']].isnull().sum().values
print("#Daily tests null values: ", nulls_in_df[0])
print("#Cases null values: ", nulls_in_df[1])
print("#Deaths null values: ", nulls_in_df[2])


# Plot Daily tests, Cases and Deaths through time to see what method of interpolation to use
grouped_df = df.groupby(pd.Grouper(key='Date', freq='D')).agg({'Daily tests': 'sum'}).reset_index()

plt.figure(figsize=(10, 6))
plt.plot(grouped_df['Date'], grouped_df['Daily tests'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Sum of Daily Tests')
plt.title('Sum of Daily Tests over Time')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


grouped_df = df.groupby(pd.Grouper(key='Date', freq='D')).agg({'Cases': 'sum'}).reset_index()

plt.figure(figsize=(10, 6))
plt.plot(grouped_df['Date'], grouped_df['Cases'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.title('Total Cases over Timer')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

grouped_df = df.groupby(pd.Grouper(key='Date', freq='D')).agg({'Deaths': 'sum'}).reset_index()

plt.figure(figsize=(10, 6))
plt.plot(grouped_df['Date'], grouped_df['Deaths'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Total Deaths')
plt.title('Total Deaths over Timer')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# We see that the data follow a curve, resembling a parabola, so we will use quadratic method to interpolate
df[['Daily tests', 'Cases', 'Deaths']] = df[['Daily tests', 'Cases', 'Deaths']].interpolate(method='quadratic', limit_direction='forward')

df[['Daily tests', 'Cases', 'Deaths']] = df[['Daily tests', 'Cases', 'Deaths']].fillna(method='ffill')

nulls_in_df = df[['Daily tests', 'Cases', 'Deaths']].isnull().sum().values
print("#Daily tests null values: ", nulls_in_df[0])
print("#Cases null values: ", nulls_in_df[1])
print("#Deaths null values: ", nulls_in_df[2])

# We can plot again a heatmap diagram to see if there has been any change between
# the relationship of the variables after processing the data

# **********************************************************************************************************************

# Clustering

# Calculate the percentages we will use as features for clustering (Positivity rate, Death rate, Cases/Population)
df['Summed tests'] = df.groupby('Entity')['Daily tests'].cumsum()

df_percentages = pd.DataFrame(columns=["Country", "Positivity Rate", "Death Rate", "Cases/Population"])
grouped_data = df.groupby("Entity")
for country, country_df in grouped_data:
    country_total_cases = country_df["Cases"].iloc[-1]
    country_total_deaths = country_df["Deaths"].iloc[-1]
    country_total_daily_tests = country_df["Summed tests"].iloc[-1]
    country_population = country_df["Population"].iloc[0]

    positivity_rate = country_total_cases / country_total_daily_tests
    death_rate = country_total_deaths / country_total_cases
    cases_population = country_total_cases / country_population

    df_percentages.loc[len(df_percentages) + 1] = [country, positivity_rate, death_rate, cases_population]

# Display the first 5 rows of the new dataframe
print([["Starting Dataframe:"]])
print(df_percentages.head())

# For preprocessing the dataset we will use RobustScaler(), because it is robust to outliers as it uses the
# interquartile range. After removing the outliers using RobustScaler(), we can use either StandardScaler() or
# MinMaxScaler(). We will use MinMaxScaler() to also bring the data into [0,1] range.
columns_to_scale = ["Positivity Rate", "Death Rate", "Cases/Population"]

robust_scaler = RobustScaler()
cluster_data = robust_scaler.fit_transform(df_percentages[columns_to_scale])
df_percentages[columns_to_scale] = cluster_data
print("Data after RobustScaler()\n:", df_percentages.head())

minMax_scaler = MinMaxScaler()
cluster_data = minMax_scaler.fit_transform(df_percentages[columns_to_scale])
df_percentages[columns_to_scale] = cluster_data
print("Data after MinMaxScaler():\n", df_percentages.head())

# check again neighbors and best distance
# After experimenting with different number of neighbors we define n_neighbors=4 as it has a smoother curve
neighbors = NearestNeighbors(n_neighbors=4)
neighbors.fit(cluster_data)
distances, indices = neighbors.kneighbors(cluster_data)

kDistances = distances[:, -1]
sorted_KDistances = np.sort(kDistances)

# Create the figure
plt.figure(figsize=(10, 6))
plt.plot(range(len(sorted_KDistances)), sorted_KDistances, linestyle='-', color='blue')
plt.title('Density Reachability Plot')
plt.xlabel('Data Points')
plt.ylabel('4th Neighbor Distance')
plt.grid(True)
plt.tight_layout()
plt.show()

# We can see that a good distance is at point 75, with 4th Neighbor distance = 0.068 ~ 0.07 so we will use that as eps
eps_neigh = 0.07

# Minimum number of samples in a neighborhood to form a core point
min_samples = 3

dbscan = DBSCAN(eps=eps_neigh, min_samples=min_samples)
clusters = dbscan.fit_predict(cluster_data)
df_percentages['Cluster'] = clusters

# Get the unique cluster labels
unique_clusters = set(clusters)

table = []
for cluster in unique_clusters:
    cluster_countries = df_percentages[df_percentages['Cluster'] == cluster]['Country']
    table.append(["Cluster id : " + str(cluster) + " Number of Countries" + ":" + str(len(cluster_countries)), ', '.join(cluster_countries)])

print("Table:\n", table)

# Evaluate the model
# The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters.
# The silhouette scores range from -1 to 1, where a higher value indicates that the object is better matched
# to its own cluster, and worse matched to neighboring clusters.

# Since we used DBSCAN we will need to exlude the noise points from the evaluation
valid_cluster = clusters[clusters != -1]
valid_data = df_percentages[df_percentages['Cluster'] != -1][['Positivity Rate', 'Death Rate', 'Cases/Population']].values
silhouette_score = silhouette_score(valid_data, valid_cluster)
print("Silhouette Score: ", silhouette_score)
