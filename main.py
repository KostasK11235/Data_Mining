import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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
# df_filtered.to_csv('interpolated_data.csv', index=False)
# min-max scaling to data
grouped_data = df_filtered.groupby('Entity')
country_Hospital_beds_per1000 = grouped_data['Hospital beds per 1000 people'].first()
country_doctors_per1000 = grouped_data['Medical doctors per 1000 people'].first()
country_population = grouped_data['Population'].first()
country_total_daily_tests = grouped_data['Daily tests'].sum()
country_total_cases = grouped_data['Cases'].max()
country_total_deaths = grouped_data['Deaths'].max()

country_Hospital_beds = country_Hospital_beds_per1000*(country_population//1000)
country_doctors = country_doctors_per1000*(country_population//1000)

positivity_rate = country_total_cases/country_total_daily_tests
death_rate = country_total_deaths/country_total_cases
cases_population = country_total_cases/country_population
beds_cases = country_Hospital_beds/country_total_cases
doctors_cases = country_doctors/country_total_cases

# get all percentages in a single Dataframe
concatenated_percentages = pd.concat([positivity_rate, death_rate, cases_population, beds_cases, doctors_cases], axis=1)

# print("Percentages: \n", concatenated_percentages)
print(concatenated_percentages.shape)
# print(type(concatenated_percentages))

X = concatenated_percentages.iloc[:, :5]
y = concatenated_percentages.index

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X_scaled)

oversample = SMOTE()
X_oversampled, y_oversampled = oversample.fit_resample(X_minmax, y)

# print(X_oversampled)
best_score = -1
best_k = -1

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_oversampled)
    score = silhouette_score(X_oversampled, labels)
    print("score: ", score, ", k=", k)
    if score > best_score:
        best_score = score
        best_k = k

print("best k", best_k)
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_oversampled)

print(labels)
concatenated_percentages['Cluster'] = labels
cluster_groups = concatenated_percentages.groupby('Cluster')
print(concatenated_percentages)
for name, group in cluster_groups:
    print("Group:", name)
    print(group)
    print("\n")


# SVM regressor
df_filtered = df_filtered.loc[df_filtered['Entity'] == 'Greece'].copy()
df_filtered.loc[:, 'positivity_rate'] = df_filtered['Cases']/df_filtered['Daily tests']
print(df_filtered)
df_filtered['Swifted positivity rate'] = df_filtered['positivity_rate'].shift(-3)

df_filtered['Swifted positivity rate'] = df_filtered['Swifted positivity rate'].interpolate(method='linear', limit_direction='forward')
df_filtered['Swifted positivity rate'] = df_filtered['Swifted positivity rate'].round()

columns_to_scale = ['Latitude', 'Longitude', 'Average temperature per year', 'Hospital beds per 1000 people',
                    'Medical doctors per 1000 people', 'GDP/Capita', 'Population', 'Median age', 'Population aged 65 and over (%)',
                    'Daily tests', 'Cases', 'Deaths', 'Swifted positivity rate']
df_filtered[columns_to_scale] = scaler.fit_transform(df_filtered[columns_to_scale])
df_filtered[columns_to_scale] = minmax.fit_transform(df_filtered[columns_to_scale])


# split the df_filtered into dates before and after 01-01-2021
train_data = df_filtered[df_filtered['Date'] < '2021-01-01'].copy()
test_data = df_filtered[df_filtered['Date'] >= '2021-01-01'].copy()

X_train = train_data[['Hospital beds per 1000 people', 'Medical doctors per 1000 people', 'Population', 'Median age',
                      'Daily tests', 'Cases', 'Deaths']]
y_train = train_data['Swifted positivity rate']

X_eval = test_data[['Hospital beds per 1000 people', 'Medical doctors per 1000 people', 'Population', 'Median age',
                    'Daily tests', 'Cases', 'Deaths']]
y_eval = test_data['Swifted positivity rate']


# create and train the SVM regressor
svm_regressor = SVR()

param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

# create the grid search object
grid_search = GridSearchCV(svm_regressor, param_grid, scoring='neg_mean_squared_error', cv=5)

# fit the grid search to the training data
grid_search.fit(X_train, y_train)

# get the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters: ", best_params)

# get the best model
# best_model = grid_search.best_estimator_
# make predictions
# predictions = best_model.predict(X_eval)

best_svm_regressor = SVR(**best_params)
best_svm_regressor.fit(X_train, y_train)

y_pred = []
for i in range(len(X_eval)):
    # make predictions for a sample
    X_sample = X_eval.iloc[[i]].values.reshape(1, -1)
    y_sample = y_eval.iloc[i]
    y_pred.append(best_svm_regressor.predict(X_sample))

    # add the predicted sample to the training set
    X_train = X_train.append(X_eval.loc[i], ignore_index=False)
    y_train = y_train.append(y_eval.loc[i], ignore_index=False)

    # retrain the SVM regressor
    best_svm_regressor.fit(X_train, y_train)

print("predictions: ", y_pred)
print("true values: ", y_eval)

# evaluate the model
mse = mean_squared_error(y_eval, y_pred)
print("Mean Squared Error: ", mse)