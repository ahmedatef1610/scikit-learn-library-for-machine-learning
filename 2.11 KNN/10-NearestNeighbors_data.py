#Import Libraries
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

#----------------------------------------------------
# reading data
data = pd.read_csv('path/2.11 KNN/data.csv')
# data.describe()
# X Data
X = data
print(X.head())
print('X shape is ' , X.shape)
# ---------
# X_train, X_test, y_train, y_test = train_test_split(X,[], test_size=0.33, random_state=44, shuffle=True)
X_train = X[:1000]
X_test  = X[1000:]
# y_train = y[:1000]
# y_test  = y[1000:]
print(X_train.shape,X_test.shape)
# print(y_train.shape,y_test.shape)
print("="*25)
#----------------------------------------------------
# Applying NearestNeighborsModel Model 
NearestNeighborsModel = NearestNeighbors(n_neighbors=4,radius=1.0,algorithm='auto')
NearestNeighborsModel.fit(X)
#----------------------------------------------------
#Calculating Details
print('NearestNeighborsModel Train kneighbors are : ' , NearestNeighborsModel.kneighbors(X_train[: 1]))
print("="*5)
print('NearestNeighborsModel Train radius kneighbors are : ' , NearestNeighborsModel.radius_neighbors(X_train[:  1]))
print("="*25)
print('NearestNeighborsModel Test kneighbors are : ' , NearestNeighborsModel.kneighbors(X_test[: 1]))
print("="*5)
print('NearestNeighborsModel Test  radius kneighbors are : ' , NearestNeighborsModel.radius_neighbors(X_test[:  1]))
print("="*25)
#----------------------------------------------------

