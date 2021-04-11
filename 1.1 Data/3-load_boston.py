# Import Libraries
from sklearn.datasets import load_boston
#----------------------------------------------------

# load boston data

BostonData = load_boston()

# X Data
X = BostonData.data
print('X Data is ' , X[:10])
print('X shape is ' , X.shape)
print('X Features are ' , BostonData.feature_names)
# print(BostonData.DESCRstr)

# y Data
y = BostonData.target
print('y Data is ' , y[:10])
print('y shape is ' , y.shape)