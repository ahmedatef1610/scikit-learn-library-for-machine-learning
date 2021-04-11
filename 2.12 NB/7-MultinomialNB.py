from sklearn.naive_bayes import MultinomialNB
import numpy as np
# ----------------------------------------------------
X = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])
# ----------------------------------------------------
clf = MultinomialNB()
clf.fit(X, y)
# ----------------------------------------------------
print(clf.score(X,y))
print(clf.predict(X[2:3]))



