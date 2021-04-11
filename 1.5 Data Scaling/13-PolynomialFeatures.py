# Import Libraries
from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
# ----------------------------------------------------
'''

class sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=True, order='C')


'''
# ----------------------------------------------------
# Polynomial the Data

X ,y = make_regression(n_samples=100, n_features=2, shuffle=True)
# showing data
print('X \n', X[:5])
# print('y \n', y[:5])

scaler = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X = scaler.fit_transform(X)

# showing data
print('X \n', X[:5])
# print('y \n', y[:5])
