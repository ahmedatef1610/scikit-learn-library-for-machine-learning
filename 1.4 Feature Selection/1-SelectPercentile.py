# Import Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
# ----------------------------------------------------
'''
class sklearn.feature_selection.SelectPercentile(score_func=<function f_classif>, percentile=10)
---
score_func callable, default=f_classif
Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with scores. 
Default is f_classif (see below “See Also”). The default function only works with classification tasks.
New in version 0.18.
---
percentileint, default=10
Percent of features to keep.

'''
# ----------------------------------------------------

# load breast cancer data

BreastData = load_breast_cancer()

# X Data
X = BreastData.data
# print('X Data is ', X[:10])
# print('X shape is ', X.shape)
print('X Features are ', BreastData.feature_names)
print()

# y Data
y = BreastData.target
# print('y Data is ', y[:10])
# print('y shape is ', y.shape)
print('y Columns are ', BreastData.target_names)
print()

# ----------------------------------------------------
# Feature Selection by Percentile
# print('Original X Shape is ' , X.shape)
FeatureSelection = SelectPercentile(score_func=f_classif, percentile=10)  # score_func can = f_classif
X = FeatureSelection.fit_transform(X, y)

# showing X Dimension
# print('X Shape is ' , X.shape)
print('Selected Features are : ', FeatureSelection.get_support())
print()
print(BreastData.feature_names[FeatureSelection.get_support()])
# ['mean perimeter' 'mean area' 'area error' 'worst radius' 'worst perimeter' 'worst area'] => chi2 20%
# ['mean perimeter' 'mean concave points' 'worst radius' 'worst perimeter' 'worst area' 'worst concave points'] => f_classif
# ['mean area' 'area error' 'worst area'] => chi2 10%
# ['mean concave points' 'worst perimeter' 'worst concave points'] => f_classif 10%