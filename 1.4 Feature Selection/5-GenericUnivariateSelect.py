# Import Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import chi2, f_classif
# ----------------------------------------------------
'''
class sklearn.feature_selection.GenericUnivariateSelect(score_func=<function f_classif>, mode='percentile', param=1e-05)
---
score_func callable, default=f_classif
Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues). 
For modes ‘percentile’ or ‘k_best’ it can return a single array scores.
---
mode{‘percentile’, ‘k_best’, ‘fpr’, ‘fdr’, ‘fwe’}, default=’percentile’
Feature selection mode.
---
paramfloat or int depending on the feature selection mode, default=1e-5
Parameter of the corresponding mode.

'''
# ----------------------------------------------------
# Feature Selection by Generic
# score_func can = f_classif : mode can = percentile,fpr,fdr,fwe

BreastData = load_breast_cancer()
X = BreastData.data
y = BreastData.target
print('Original X Shape is ' , X.shape)

FeatureSelection = GenericUnivariateSelect(score_func=chi2, mode='k_best', param=3)
# FeatureSelection = GenericUnivariateSelect(score_func=chi2, mode='percentile', param=10)
X = FeatureSelection.fit_transform(X, y)

# showing X Dimension
print('X Shape is ' , X.shape)
print('Selected Features are : ' , FeatureSelection.get_support())
print(BreastData.feature_names[FeatureSelection.get_support()])

# ['mean area' 'area error' 'worst area'] => score_func=chi2, mode='k_best', param=3
# ['mean area' 'area error' 'worst area'] => score_func=chi2, mode='percentile', param=10 -> 10%