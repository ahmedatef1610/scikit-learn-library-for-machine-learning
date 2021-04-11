from sklearn.naive_bayes import BernoulliNB
import numpy as np
# ----------------------------------------------------
X = np.random.randint(100, size=(10000, 100))
Y = np.random.randint(5, size=(10000, 1)).flatten()

print(X.shape)
print(Y.shape)
# ----------------------------------------------------
clf = BernoulliNB()
clf.fit(X, Y)
print(clf.score(X,Y))
# ----------------------------------------------------
Z = np.random.randint(10, size=(1, 100))
print(clf.predict(Z))
