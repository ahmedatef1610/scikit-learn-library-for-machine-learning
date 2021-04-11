
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.arange(6).reshape(3, 2)
print(X)

poly = PolynomialFeatures(degree=2, include_bias=True)
X1 = poly.fit_transform(X)
print(X1)


poly = PolynomialFeatures(interaction_only=True)
X2 = poly.fit_transform(X)
print(X2)
