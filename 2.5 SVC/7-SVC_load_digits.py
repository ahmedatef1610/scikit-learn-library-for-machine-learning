import matplotlib.pyplot as plt
from sklearn import datasets, svm
# ----------------------------------------------------
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
# ----------------------------------------------------
x_train = X_digits[:-100]
x_test = X_digits[-100:]
y_train = y_digits[:-100]
y_test = y_digits[-100:]
# ----------------------------------------------------
svc = svm.SVC(kernel='linear', C=1)
svc.fit(x_train, y_train)
# -----------------
sc = svc.score(x_test, y_test)
print('score = ', sc)
print("="*25)
# ----------------------------------------------------
for j in range(1, 100, 20):
    print('Prediction  is ', svc.predict(X_digits[j].reshape(1, -1)))
    plt.imshow(digits.images[j], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
