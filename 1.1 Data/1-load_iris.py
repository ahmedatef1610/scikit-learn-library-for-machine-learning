# Import Libraries
from sklearn.datasets import load_iris
#----------------------------------------------------

# load iris data

IrisData = load_iris()

# X Data
X = IrisData.data
print('X Data is ' , X[:10])
print('X shape is ' , X.shape)
print('X Features are ' , IrisData.feature_names)

# y Data
y = IrisData.target
print('y Data is ' , y[:10])
print('y shape is ' , y.shape)
print('y Columns are ' , IrisData.target_names)