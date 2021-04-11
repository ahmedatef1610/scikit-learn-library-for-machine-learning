from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, chi2
import numpy as np

DigitsData = load_digits()
print('X Features are ', DigitsData.feature_names)
print('y Columns are ', DigitsData.target_names)

X, y = DigitsData.data, DigitsData.target
print(X.shape)

FeatureSelection = SelectPercentile(score_func=chi2, percentile=10)
X_new = FeatureSelection.fit_transform(X, y)
print(X_new.shape)
print(np.array(DigitsData.feature_names)[FeatureSelection.get_support()])
