# Importing the libraries
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ----------------------------------------------------
dataset = pd.read_csv('path/2.4 SVR/Earthquakes.csv')
# print(dataset.head())
# print("="*10)
# print(dataset.info())
# print("="*10)
# print(dataset.describe())
# print("="*10)
# ----------------
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(dataset)
dataset = imp.transform(dataset)
# ----------------
sc = StandardScaler()
dataset = sc.fit_transform(dataset)
# ----------------
X = dataset[:, :-1]
y = dataset[:, -1]
print(X.shape)
print(y.shape)
print("="*10)
# ----------------------------------------------------
clf = SVR(kernel = 'rbf')
clf.fit(X, y)
# ----------------------------------------------------
print("score : ",clf.score(X,y)) 
print("="*10)
# 0.8227470273341159 => rbf
# ---------------------
y_pred = clf.predict([[90,12,12,-5,54,0.3,0.9,3.5,16.2,10]]) 
print(y_pred)
print("="*5)
y_pred = clf.predict([X[16]])
print(y[16] , ' => ' , y_pred)
print("="*10)
# ----------------------------------------------------
