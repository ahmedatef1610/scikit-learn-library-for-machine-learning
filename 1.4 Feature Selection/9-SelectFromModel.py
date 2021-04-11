# Import Libraries
from sklearn.feature_selection import SelectFromModel
# ----------------------------------------------------
'''
class sklearn.feature_selection.SelectFromModel(estimator, threshold=None, prefit=False, norm_order=1, max_features=None,
importance_getter='auto')
---


'''
'''
# from sklearn.linear_model import LinearRegression
# thismodel = LinearRegression()
'''
# ----------------------------------------------------
# Feature Selection by KBest

from sklearn.datasets import load_breast_cancer
BreastData = load_breast_cancer()
X = BreastData.data
y = BreastData.target
print('Original X Shape is ', X.shape)

from sklearn.linear_model import LinearRegression
thismodel = LinearRegression()
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)

# make sure that thismodel is well-defined
FeatureSelection = SelectFromModel(estimator=thismodel, max_features=None)
X = FeatureSelection.fit_transform(X, y)

# showing X Dimension
print('X Shape is ', X.shape)
print('Selected Features are : ', FeatureSelection.get_support())
print(BreastData.feature_names[FeatureSelection.get_support()])
