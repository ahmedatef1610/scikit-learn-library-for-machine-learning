import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# ----------------------------------------------------
X = np.random.randint(20, size=(50, 10))
y = np.random.randint(5, size=(50, 1))
# ----------------------------------------------------
clf = LDA()
clf.fit(X, y)
# ----------------------------------------------------
print(clf.score(X,y))
z = np.random.randint(20, size=(1, 10))
print(clf.predict(z))


