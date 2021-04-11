#Import Libraries
from sklearn.datasets import make_regression
from sklearn.preprocessing import Binarizer
#----------------------------------------------------
'''

class sklearn.preprocessing.Binarizer(threshold=0.0, copy=True)

threshold float, default=0.0
Feature values below or equal to this are replaced by 0, above it by 1. 
Threshold may not be less than 0 for operations on sparse matrices.

'''
# ----------------------------------------------------
#Binarizing Data

X ,y = make_regression(n_samples=500, n_features=3,shuffle=True)
# showing data
print('X \n', X[:5])
print('y \n', y[:5])

scaler = Binarizer(threshold = 1.0)
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:5])
print('y \n' , y[:5])