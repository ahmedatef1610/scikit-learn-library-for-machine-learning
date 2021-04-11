#Import Libraries
from sklearn.datasets import load_wine
#----------------------------------------------------

#load wine data

WineData = load_wine()

#X Data
X = WineData.data
print('X Data is ' , X[:10])
print('X shape is ' , X.shape)
print('X Features are ' , WineData.feature_names)

#y Data
y = WineData.target
print('y Data is ' , y[:10])
print('y shape is ' , y.shape)
print('y Columns are ' , WineData.target_names)