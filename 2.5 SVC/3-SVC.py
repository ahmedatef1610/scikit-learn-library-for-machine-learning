import numpy as np
from sklearn.svm import SVC
# ----------------------------------------------------
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1,1,2,2])
# y = np.array(['a','a','b','b'])
# ----------------------------------------------------
clf = SVC(kernel='rbf',degree=3,C=10.0,gamma='scale',)
clf.fit(X, y)
# ----------------------------------------------------
print(clf.predict([[-0.8, -1]]))
print("="*10)
print("score : ",clf.score(X,y))
print("="*10)
y_pred = clf.predict([X[0]])
print(y[0] , ' => ' ,y_pred)
print("="*10)