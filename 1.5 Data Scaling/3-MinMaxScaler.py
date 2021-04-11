# Import Libraries
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
# ----------------------------------------------------
'''
MinMaxScaler   (Normalization)

class sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True, clip=False)

'''
# ----------------------------------------------------

# MinMaxScaler for Data

X ,y = make_regression(n_samples=1000, n_features=5,shuffle=True)
# showing data
print('X ', X[:5])
print('y ', y[:5])

scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)

# showing data
print('X ', X[:5])
print('y ', y[:5])
