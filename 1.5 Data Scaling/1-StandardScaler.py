# Import Libraries
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
# ----------------------------------------------------
'''
Standard Scaler    (Standardization)

class sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

'''
# ----------------------------------------------------

# Standard Scaler for Data

X ,y = make_regression(n_samples=1000, n_features=5,shuffle=True)
# showing data
print('X ', X[:5])
print('y ', y[:5])

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X = scaler.fit_transform(X)

print()
print()
print()

# showing data
print('X ', X[:5])
print('y ', y[:5])
