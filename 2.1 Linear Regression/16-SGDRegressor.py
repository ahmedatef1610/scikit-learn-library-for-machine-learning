import numpy as np
import pandas as pd
from sklearn import linear_model
# ----------------------------------------------------
dataset = pd.read_csv('path/2.1 Linear Regression/houses.csv')
dataset.head()
print(dataset.info())
print("="*10)
print(dataset.describe())
print("="*25)
# ----------------------------------------------------
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(dataset)
dataset = imp.transform(dataset)
print("="*25)
# ----------------------------------------------------
X = dataset[:, :-1]
y = dataset[:, -1]

print(X.shape)
print(y.shape)
print("="*25)
# ----------------------------------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print("="*25)
# ----------------------------------------------------
from sklearn.linear_model import SGDRegressor
#clf = linear_model.SGDRegressor( penalty = 'l2' , max_iter=1000, tol=1e-3 , loss = 'huber')
#clf = linear_model.SGDRegressor( penalty = 'l1' , max_iter=1000, tol=1e-3 , loss = 'huber')
#clf = linear_model.SGDRegressor( penalty = 'l1' , max_iter=1000, tol=1e-3 , loss = 'squared_loss')
sgd = SGDRegressor( penalty = 'l2' , max_iter=1000, tol=1e-3 , loss = 'squared_loss')
sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test) 
print("y_pred ",y_pred[:5])
print("y_test ",y_test[:5])
print(sgd.score(X_train,y_train))
print(sgd.score(X_test,y_test))
print("="*25)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred))

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))
print("="*25)

# ----------------------------------------------------
