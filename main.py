import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import LSTM, Dense
import regex as re

def findBestParameters(model, train_X, train_y):
    parameters = {
            'C':[0.0001, 0.001, 0.01, 1, 10, 20, 100],
            'epsilon': [0.1, 0.01, 0.05, 0.001],
            'gamma': [0.0001, 0.001, 0.1, 1, 10, 100, 'scale', 'auto']
            }
    clf = GridSearchCV(model, param_grid=parameters, cv=5)
    clf.fit(train_X, train_y)

    return SVR(C=clf.best_params_['C'], epsilon=clf.best_params_['epsilon'], gamma=clf.best_params_['gamma'])

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
# After experimenting with different number of neighbors we define n_neighbors=2, as the elbow curve that occurs
# gives an eps distance with better evaluation results from silhouette score
neighbors = NearestNeighbors(n_neighbors=3)
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

# We can see that a good distance is at point 70, with kth Neighbor distance ~= 0.068, near the elbow point so we will use that as eps
eps_neigh = 0.068

# Define the minimum number of samples that each core point must have in his neighborhood
min_samples = 3

dbscan = DBSCAN(eps=eps_neigh, min_samples=min_samples)
clusters = dbscan.fit_predict(cluster_data)
df_percentages['Cluster'] = clusters

# Get the unique cluster labels
formed_clusters = set(clusters)

country_clusters = []
for cluster in formed_clusters:
    clustered_country = df_percentages[df_percentages['Cluster'] == cluster]['Country']
    country_clusters.append(["Cluster id : " + str(cluster) + " Number of Countries" + ":" + str(len(clustered_country)), ', '.join(clustered_country)])

print("Table:\n", country_clusters)

# We will use silhouette score to evaluate the quality of the clusters.
# Score near +1 indicates that the object is well-matched to its own cluster and poorly-matched to neighboring clusters.
# Score near 0 indicates that the object is on or very close to the decision boundary between two neighboring clusters.
# Score near -1 indicates that the object is better-matched to a neighboring cluster than to its own cluster.
# In general, a higher silhouette score suggests that the clustering is appropriate and the clusters are well-defined and distinct

# To evaluate the formed clusters we ignore the noise points
valid_clusters = clusters[clusters != -1]
valid_data = df_percentages[df_percentages['Cluster'] != -1][['Positivity Rate', 'Death Rate', 'Cases/Population']].values
silhouette_score = silhouette_score(valid_data, valid_clusters)
print("Silhouette Score: ", silhouette_score)

# Assuming your df_percentages DataFrame contains columns 'Positivity Rate', 'Death Rate', and 'Cases/Population'
filtered_df = df_percentages[df_percentages['Cluster'] != -1]
x = filtered_df['Positivity Rate']
y = filtered_df['Death Rate']
z = filtered_df['Cases/Population']
c = filtered_df['Cluster']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the scatter plot
scatter = ax.scatter(x, y, z, c=c, cmap='viridis')

# Customize the plot
ax.set_xlabel('Positivity Rate')
ax.set_ylabel('Death Rate')
ax.set_zlabel('Cases/Population')
ax.set_title('Clustered Countries')

# Add a color bar
cbar = fig.colorbar(scatter)
cbar.set_label('Cluster')

plt.show()

# **********************************************************************************************************************

# SVR

# We need only the data for Greece. First we choose only the records with entity Greece and then
# convert into datetime column Date

df_greece = df[df['Entity'] == 'Greece'].reset_index(drop=True)
df_greece['Date'] = pd.to_datetime(df_greece['Date'])
greece_population = df_greece['Population'].iloc[0]

features_to_drop = ['Entity', 'Continent', 'Latitude', 'Longitude', 'Average temperature per year',
                    'Hospital beds per 1000 people', 'Medical doctors per 1000 people', 'GDP/Capita',
                    'Population', 'Median age', 'Population aged 65 and over (%)', 'Deaths', 'Summed tests']

df_greece = df_greece.drop(features_to_drop, axis=1)
print(df_greece)

# we will convert the 'Date' column of the DataFrame to Unix timestamps
starting_date = df_greece.loc[0, 'Date']
df_greece['Days'] = (df_greece['Date'] - starting_date).dt.days
df_greece['Timestamp'] = df_greece['Date'].values.astype(np.int64) // 10 ** 9
df_greece[['Daily tests', 'Cases']] = df_greece[['Daily tests', 'Cases']].abs()
df_greece['Positivity rate'] = df_greece['Cases'].diff() / df_greece['Daily tests']

# we drop the first row, Date: 2020-03-30, because Date: 2020-03-31 is missing
df_greece = df_greece.drop(0).reset_index(drop=True)

# We plot the positivity rate in Greece through the dates in df_greece
plt.figure(figsize=(12, 5))
plt.plot(df_greece['Date'], df_greece['Positivity rate'])
plt.title('Positivity rate in Greece though the registered dates')
plt.xlabel('Date')
plt.ylabel('Positivity Rate')
plt.show()

# drop dates >= 2021-02-24 and 2020-04-02
df_greece = df_greece[df_greece['Date'] < pd.to_datetime('2021-02-23')]
df_greece = df_greece.drop(0).reset_index(drop=True)

# We plot the positivity rate in Greece through the dates in df_greece
plt.figure(figsize=(12, 5))
plt.plot(df_greece['Date'], df_greece['Positivity rate'])
plt.title('Positivity rate in Greece though the registered dates')
plt.xlabel('Date')
plt.ylabel('Positivity Rate')
plt.show()

# To train the regressor we will use all the dates before 2021-01-01
train_dates = df_greece[df_greece['Date'] < datetime(2021, 1, 1)]['Timestamp'].values.reshape(-1, 1)
train_positivity = df_greece[df_greece['Date'] < datetime(2021, 1, 1)]['Positivity rate'].values.reshape(-1, 1)

scaler = StandardScaler()
train_positivity_scaled = scaler.fit_transform(train_positivity)

predicted_positivity = []

model = SVR()
model = findBestParameters(model, train_dates, train_positivity_scaled.ravel())

# Predict the remaining values ( But the last three )
remaining_dates = df_greece[df_greece['Date'] >= datetime(2021, 1, 1)]['Timestamp'][:-3]
remaining_dates = remaining_dates.reset_index(drop=True)
for today in remaining_dates:
    train_dates = np.append(train_dates, today).reshape(-1, 1)
    # Gather the positivity rate found today
    train_positivity = np.append(train_positivity, df_greece[df_greece['Timestamp'] == today]['Positivity rate'].iloc[0])
    train_positivity = train_positivity.reshape(-1, 1)

    # Scale the new data
    scaler = StandardScaler()
    train_positivity_scaled = scaler.fit_transform(train_positivity)

    # Create a new SVR model for each iteration and fit it with the training data
    #svr = SVR()
    #svr.fit(train_dates, train_positivity_scaled.ravel())
    model.fit(train_dates, train_positivity_scaled.ravel())

    # calculate the timestamp for the prediction date (+3 days)
    predict_ts = today + 3 * 24 * 60 * 60

    # Predict the positivity rate for the prediction date
    predicted_value_scaled = model.predict([[predict_ts]])
    predicted_value = scaler.inverse_transform(predicted_value_scaled.reshape(-1, 1))

    # # Store the predicted value
    predicted_positivity.append(predicted_value[0][0])

# Calculate the MSE
predicted_pos_rate_values = predicted_positivity[:-3]
dates = df_greece[df_greece['Date'] >= datetime(2021, 1, 1)]['Date'][3:-3].to_list()
real_pos_rate_values = df_greece[df_greece['Date'] >= datetime(2021, 1, 1)]['Positivity rate'][3:-3].to_list()
mse = mean_squared_error(real_pos_rate_values, predicted_pos_rate_values)
print('MSE: ', mse)
print('Real values: ', real_pos_rate_values)
print('Pred values: ', predicted_pos_rate_values)

# # Plot the predicted values against the real values
plt.figure(figsize=(12, 5))
plt.plot(dates, predicted_pos_rate_values, label='Predicted Values')
plt.plot(dates, real_pos_rate_values, label='Real Values')
plt.title('Positivity Rate Prediction')
plt.xlabel('Date')
plt.ylabel('Positivity Rate')
plt.legend()
plt.show()

# RNN

# We already have the train_dates and train_pos_rate, we just need to reshape them for the tensor
reshaped_train_dates = np.reshape(train_dates, (train_dates.shape[0], train_dates.shape[1], 1))
reshaped_train_positivity_scaled = np.reshape(train_positivity_scaled, (train_positivity_scaled.shape[0], train_positivity_scaled.shape[1], 1))

# Create a Sequential model
model = Sequential()

# Add LSTM layers
model.add(LSTM(units=70, input_shape=(train_dates.shape[1], 1)))
model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(train_dates, train_positivity_scaled, epochs=16, batch_size=5)

plt.figure(figsize=(6, 5))
plt.plot(history.history['loss'])
plt.title('Model Loss for Initial Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

rnn_predictions = []

for today in remaining_dates:
    # Gather the positivity rate found today
    new_train_pos_rate = df_greece[df_greece['Timestamp'] == today]['Positivity rate'].iloc[0]

    # Scale the new data
    new_train_pos_rate_scaled = scaler.transform([[new_train_pos_rate]])

    # LSTM requires the data to be in a specific format, 3d tensor
    today_tensor = np.reshape(today, (1, 1, 1))
    new_train_pos_rate_scaled = np.reshape(new_train_pos_rate_scaled, (1, 1, 1))

    # Train the model with the new data
    model.fit(today_tensor, new_train_pos_rate_scaled, epochs=16, batch_size=3)

    # calculate the prediction date
    predict_ts = today + 3 * 24 * 60 * 60
    predict_ts = np.reshape(predict_ts, (1, 1, 1))

    # Predict the positivity rate for the prediction date
    predicted_value_scaled = model.predict(predict_ts, verbose=0)
    predicted_value = scaler.inverse_transform(predicted_value_scaled.reshape(-1, 1))

    # Store the predicted value
    rnn_predictions.append(predicted_value[0][0])

# Calculate the MSE
rnn_predictions = rnn_predictions[:-3]
dates = df_greece[df_greece['Date'] >= datetime(2021, 1, 1)]['Date'][3:-3].to_list()
real_pos_rate_values = df_greece[df_greece['Date'] >= datetime(2021, 1, 1)]['Positivity rate'][3:-3].to_list()
mse = mean_squared_error(real_pos_rate_values, rnn_predictions)
print('MSE: ', mse)
print('Real values: ', real_pos_rate_values)
print('Pred values: ', rnn_predictions)

# # Plot the predicted values against the real values
plt.figure(figsize=(12, 5))
plt.plot(dates, rnn_predictions, label='Predicted Values')
plt.plot(dates, real_pos_rate_values, label='Real Values')
plt.title('Positivity Rate Prediction')
plt.xlabel('Date')
plt.ylabel('Positivity Rate')
plt.legend()
plt.show()
