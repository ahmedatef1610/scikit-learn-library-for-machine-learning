import pandas as pd
# ----------------------------------------------------
dataset = pd.read_csv('path/2.2 Logistic Regression/heart.csv')
dataset.head(20)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
print(X.head())
print(y.head())
# ----------------------------------------------------
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# ----------------------------------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print(X_train)
# print(X_test)
# ----------------------------------------------------
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='log', penalty='l2', max_iter=10000, tol=1e-5)
sgd.fit(X_train, y_train)
print(sgd.n_iter_)

# Predicting the Test set results
y_pred = sgd.predict(X_test)
print(y_pred[:5])
#probability of all values
pr = sgd.predict_proba(X_test)
print(pr[:5])
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))
from sklearn.metrics import median_absolute_error
print(median_absolute_error(y_test, y_pred))
from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred, average='micro'))
print(sgd.score(X_test, y_test))

 