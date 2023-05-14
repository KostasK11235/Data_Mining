import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# read data file
df = pd.read_csv('data.csv')

mean_values = df.groupby('Entity')[['Daily tests', 'Cases', 'Deaths']].mean()
print(mean_values)

var_values = df.groupby('Entity')[['Daily tests', 'Cases', 'Deaths']].var()
print(var_values)

# beds vs deaths
beds = df.groupby('Hospital beds per 1000 people')['Deaths'].sum()
beds.plot(kind='bar')
plt.xlabel('Hospital beds per 1000 people')
plt.ylabel('Deaths')
plt.show()

# doctors vs deaths
md_deaths = df.groupby('Medical doctors per 1000 people')['Deaths'].sum()
md_deaths.plot(kind='bar')
plt.xlabel('Medical Doctors per 1000 people')
plt.ylabel('Deaths')
plt.show()

# GDP/capita vs doctors
gdp_deaths = df.groupby('GDP/Capita')['Medical doctors per 1000 people'].mean()
gdp_deaths.plot(kind='bar')
plt.xlabel('GDP/Capita')
plt.ylabel('Doctors')
plt.show()