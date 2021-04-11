from sklearn.svm import SVR
import numpy as np
# ----------------------------------------------------
n_features = 10
n_samples=10
np.random.seed(0)
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)
print(X.shape)
print(y)
# ----------------------------------------------------
clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
clf.fit(X, y)
# ----------------------------------------------------
# newX = np.random.randn(1,10)
# y_pred = clf.predict(newX)
# print(newX , ' \n ' ,y_pred)
print("score : ",clf.score(X,y))
y_pred = clf.predict([X[0]])
print(y[0] , ' => ' ,y_pred)