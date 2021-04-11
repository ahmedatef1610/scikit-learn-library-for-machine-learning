from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, chi2
import numpy as np

X, y = load_digits(return_X_y=True)
print(X.shape)

X_new = SelectPercentile(score_func=chi2, percentile=10)
X_new.fit(X, y)
selected = X_new.transform(X)
print(X_new.get_support())

result = np.where(X_new.get_support() == True)
print(X_new.get_support()[result])
print(X_new.get_support()[X_new.get_support()==True])
print(X_new.get_support()[result].__len__())
