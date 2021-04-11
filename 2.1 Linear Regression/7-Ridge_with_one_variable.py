# Import Libraries
from sklearn.linear_model import Ridge

from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
'''
sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, 
solver='auto', random_state=None)


'''
# ----------------------------------------------------

X, y = make_regression(n_samples=10000, n_features=1, shuffle=True, noise=25, random_state=16)
print(X.shape,y.shape)
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)

print(X.shape)
print(y.shape)
print("="*10)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print("="*25)

# Applying Ridge Regression Model
RidgeRegressionModel = Ridge(alpha=1.0, random_state=33)
RidgeRegressionModel.fit(X_train, y_train)

# Calculating Details
print('Ridge Regression Train Score is : ', RidgeRegressionModel.score(X_train, y_train))
print('Ridge Regression Test Score is : ', RidgeRegressionModel.score(X_test, y_test))
print('Ridge Regression Coef is : ', RidgeRegressionModel.coef_)
print('Ridge Regression intercept is : ', RidgeRegressionModel.intercept_)
print('----------------------------------------------------')

# Calculating Prediction
y_pred = RidgeRegressionModel.predict(X_test)
print('Pred Value for Ridge Regression is : ', y_pred[:5])
print('True Value for Ridge Regression is : ', y_test[:5])
# ----------------------------------------------------
x_axis = np.arange(0,1,0.001)
x_axis = x_axis.reshape(-1,1)
print(x_axis.shape)

sns.scatterplot(x=X[:,0], y=y,alpha=0.5);
# sns.lineplot(x=x_axis[:,0], y=LinearRegressionModel.predict(x_axis));
sns.lineplot(x=X[:,0], y=RidgeRegressionModel.predict(X));
plt.show()