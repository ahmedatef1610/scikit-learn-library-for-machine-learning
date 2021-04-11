
# Importing the libraries
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
# ----------------------------------------------------
dataset = pd.read_csv('path/2.4 SVR/houses.csv')
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
# ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*10)
# ----------------------------------------------------
from sklearn.svm import SVR
clf = SVR(kernel = 'rbf')
clf.fit(X_train, y_train)
# ----------------------------------------------------
print('clf Train Score is : ', clf.score(X_train, y_train))
print('clf Test Score is : ', clf.score(X_test, y_test))
print("="*10)
# ----------------------------------------------------
y_pred = clf.predict(X_test) 
print('Pred Value for SVRModel is : ', y_pred[:5])
print('True Value for SVRModel is : ', y_test[:5])
print("="*10)
# ----------------------------------------------------
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred))

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))
# ----------------------------------------------------
