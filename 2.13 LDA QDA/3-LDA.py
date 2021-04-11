import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# ----------------------------------------------------
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
# ----------------------------------------------------
clf = LDA()
clf.fit(X, y)
# ----------------------------------------------------
print('LDAModel Score is : ', clf.score(X,y))
print(clf.predict([[-0.8, -1]]))