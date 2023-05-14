import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# read data file
df = pd.read_csv('data.csv')


# GDP/Capita vs Hospital beds per 1000 people
gdp_beds = df.groupby('GDP/Capita')['Hospital beds per 1000 people'].mean()
gdp_beds.plot(kind='bar')
plt.title('GDP/Capita vs Hospital beds per 1000 people')
plt.xlabel('GDP/Capital')
plt.ylabel('Hospital beds per 1000 people')
plt.show()

# GDP/capita vs Medical doctors per 1000 people
gdp_doctors = df.groupby('GDP/Capita')['Medical doctors per 1000 people'].mean()
gdp_doctors.plot(kind='bar')
plt.title('GDP/capita vs Medical doctors per 1000 people')
plt.xlabel('GDP/Capita')
plt.ylabel('Medical doctors per 1000 people')
plt.show()

# Hospital beds per 1000 people vs Deaths
beds = df.groupby('Hospital beds per 1000 people')['Deaths'].sum()
beds.plot(kind='bar')
plt.title('Hospital beds per 1000 people vs Deaths')
plt.xlabel('Hospital beds per 1000 people')
plt.ylabel('Deaths')
plt.show()

# Medical doctors per 1000 people vs Deaths
md_deaths = df.groupby('Medical doctors per 1000 people')['Deaths'].sum()
md_deaths.plot(kind='bar')
plt.title('Medical doctors per 1000 people vs Deaths')
plt.xlabel('Medical Doctors per 1000 people')
plt.ylabel('Deaths')
plt.show()

# Population vs Daily tests
population_dailyTests = df.groupby('Population')['Daily tests'].sum()
population_dailyTests.plot(kind='bar')
plt.title('Population vs Daily tests')
plt.xlabel('Population')
plt.ylabel('Total Number of Daily tests')
plt.show()

# Populations vs Cases
population_cases = df.groupby('Population')['Cases'].sum()
population_cases.plot(kind='bar')
plt.title('Populations vs Cases')
plt.xlabel('Population')
plt.ylabel('Total number of Cases')
plt.show()

# Population vs Deaths
population_deaths = df.groupby('Population')['Deaths'].sum()
population_deaths.plot(kind='bar')
plt.title('Population vs Deaths')
plt.xlabel('Population')
plt.ylabel('Deaths')
plt.show()

# Median age vs Deaths
medianAge_deaths = df.groupby('Median age')['Deaths'].sum()
medianAge_deaths.plot(kind='bar')
plt.title('Median age vs Deaths')
plt.xlabel('Median age')
plt.ylabel('Deaths')
plt.show()

# Number of tests done each date
df['Date'] = pd.to_datetime(df['Date'])
tests_by_date = df.groupby('Date')['Daily tests'].sum()
tests_by_date.plot(kind='bar')
plt.title('Number of tests done each date')
plt.xlabel('Date')
plt.ylabel('Number of Daily tests')
plt.show()
