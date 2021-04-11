# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print("="*50)
#-------------------------------------------------------------------------------------------------
sns.set()
df = pd.read_csv('path/4 Time Series (preprocessing)/multiTimeline.csv', skiprows=1)
print(df.head())
print(df.info())
print("="*50)
#---------------------------------
df.columns = ['month', 'diet', 'gym', 'finance']
print(df.head())
df.month = pd.to_datetime(df.month)
df.set_index('month', inplace=True)
print(df.head())
#---------------------------------
df.plot(figsize=(7,4), linewidth=1, fontsize=10)
plt.xlabel('Year', fontsize=10);
plt.show()
#---------------------------------
df[['diet']].plot(figsize=(7,4), linewidth=1, fontsize=10)
plt.xlabel('Year', fontsize=10);
plt.show()
#---------------------------------
diet = df[['diet']]
diet.rolling(12).mean().plot(figsize=(7,4), linewidth=1, fontsize=10)
plt.xlabel('Year', fontsize=10);
plt.show()
#---------------------------------
gym = df[['gym']]
gym.rolling(12).mean().plot(figsize=(7,4), linewidth=1, fontsize=10)
plt.xlabel('Year', fontsize=10);
plt.show()
#---------------------------------
df_rm = pd.concat([diet.rolling(12).mean(), gym.rolling(12).mean()], axis=1)
df_rm.plot(figsize=(7,4), linewidth=1, fontsize=10)
plt.xlabel('Year', fontsize=10);
plt.show()
#---------------------------------
diet.diff().plot(figsize=(7,4), linewidth=1, fontsize=10)
plt.xlabel('Year', fontsize=10);
plt.show()
#---------------------------------
df.plot(figsize=(7,4), linewidth=1, fontsize=10)
plt.xlabel('Year', fontsize=10);
plt.show()
#---------------------------------
print(df.corr())
df.diff().plot(figsize=(7,4), linewidth=1, fontsize=10)
plt.xlabel('Year', fontsize=10);
plt.show()
print(df.diff().corr())
pd.plotting.autocorrelation_plot(diet);
plt.show()
#---------------------------------



