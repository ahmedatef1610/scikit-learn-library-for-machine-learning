# Import Libraries
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
# ----------------------------------------------------
'''
class sklearn.feature_selection.SelectKBest(score_func=<function f_classif>, k=10)
---
score_func callable, default=f_classif
Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with scores. 
Default is f_classif (see below “See Also”). The default function only works with classification tasks.
New in version 0.18.
---
k int or “all”, default=10
Number of top features to select. The “all” option bypasses selection, for use in a parameter search.
---

'''
# ----------------------------------------------------
# Feature Selection by KBest

from sklearn.datasets import load_breast_cancer
BreastData = load_breast_cancer()
X = BreastData.data
y = BreastData.target
print('Original X Shape is ', X.shape)

# score_func can = f_classif
FeatureSelection = SelectKBest(score_func=chi2, k=3)
X = FeatureSelection.fit_transform(X, y)

# showing X Dimension
print('X Shape is ', X.shape)
print('Selected Features are : ', FeatureSelection.get_support())
print(BreastData.feature_names[FeatureSelection.get_support()])
