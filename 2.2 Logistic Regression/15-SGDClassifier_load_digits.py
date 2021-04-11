from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
# ----------------------------------------------------
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
print(X_digits.shape)
print(y_digits.shape)
print("="*25)
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_digits , y_digits ,  test_size = 0.2, random_state = 0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print("="*25)
# ----------------------------------------------------
sgd = SGDClassifier(loss='log', penalty='l2', max_iter=10000, tol=1e-5, random_state=0)
sgd.fit(X_train, y_train)

print(sgd.score(X_train, y_train))
print(sgd.score(X_test, y_test))

y_pred = sgd.predict(X_test)
print(y_pred[:5])
print(y_test[:5])

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred))



