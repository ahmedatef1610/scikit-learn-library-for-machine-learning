#Import Libraries
from sklearn.datasets import make_regression
from sklearn.preprocessing import MaxAbsScaler
#----------------------------------------------------
'''
work on columns

class sklearn.preprocessing.MaxAbsScaler(copy=True)


'''
# ----------------------------------------------------
# MaxAbsScaler Data

X ,y = make_regression(n_samples=500, n_features=3,shuffle=True)
X = X*100
# showing data
print('X \n', X[:5])
print('y \n', y[:5])

scaler = MaxAbsScaler(copy=True)
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:5])
print('y \n' , y[:5])