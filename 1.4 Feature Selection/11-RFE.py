# Import Libraries
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2, f_classif
# ----------------------------------------------------
'''
# Backward Elimination(Recursive Feature Elimination)

class sklearn.feature_selection.RFE(estimator, n_features_to_select=None, step=1, verbose=0, importance_getter='auto')

'''
# ----------------------------------------------------


from sklearn.datasets import load_breast_cancer
BreastData = load_breast_cancer()
X = BreastData.data
y = BreastData.target
print('Original X Shape is ', X.shape)


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)

# make sure that thismodel is well-defined
FeatureSelection = RFE(estimator=clf, n_features_to_select=3)
X = FeatureSelection.fit_transform(X, y)

# showing X Dimension
print('X Shape is ', X.shape)
# print('Selected Features are : ', FeatureSelection.get_support())
print(BreastData.feature_names[FeatureSelection.get_support()])
