import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
# ----------------------------------------------------
X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
X = abs(X)
print(X.shape)
print(y.shape)
# ----------------------------------------------------
clf = RandomForestClassifier(n_estimators=100, max_depth=3,random_state=0)
clf.fit(X, y)
print('clf Train Score is : ' , clf.score(X, y))
# ----------------------------------------------------
print(clf.feature_importances_)
print(clf.predict([[0, 0, 0, 0]]))
# ----------------------------------------------------
for i in range(50):
    l = list(np.round(np.random.rand(4),5))
    print(l , '      ' ,np.round(clf.predict([ l ]),2))
 


 