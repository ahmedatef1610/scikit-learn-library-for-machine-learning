import pandas as pd
# -------------------------------------------------------------------
dataset = pd.read_csv('path/2.2 Logistic Regression/heart.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

print(dataset.head())
print(X[:5])
print(y[:5])
print("="*25)
# -------------------------------------------------------------------
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape) 
print("="*25)
# -------------------------------------------------------------------
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train[:5])
print(X_test[:5])
print("="*25)
# -------------------------------------------------------------------
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
clss = LogisticRegression(random_state = 0)
clss.fit(X_train, y_train)
# Predicting the Test set results
y_pred = clss.predict(X_test)
print(y_pred[:5]) 
print(y_test[:5]) 
print("="*25)

print(clss.n_iter_)
print(clss.classes_)
print("="*25)

#probability of all values
pr = clss.predict_proba(X_test)[0:5,:]
print(pr)
#probability of zeros
pr = clss.predict_proba(X_test)[0:5,0]
print(pr)
#probability of ones
pr = clss.predict_proba(X_test)[0:5,1]
print(pr)
print("="*25)
# -------------------------------------------------------------------
#  Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))
from sklearn.metrics import median_absolute_error
print(median_absolute_error(y_test, y_pred))

 

 