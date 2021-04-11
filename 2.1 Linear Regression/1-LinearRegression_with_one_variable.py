# Import Libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
'''
class sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None, positive=False)


'''
# ----------------------------------------------------
X, y = make_regression(n_samples=10000, n_features=1, shuffle=True, noise=25, random_state=16)
print(X.shape,y.shape)
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
# ----------------------------------------------------
# Applying Linear Regression Model
LinearRegressionModel = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=-1)
LinearRegressionModel.fit(X_train, y_train)

# Calculating Details
print('Linear Regression Train Score is : ', LinearRegressionModel.score(X_train, y_train))
print('Linear Regression Test Score is : ', LinearRegressionModel.score(X_test, y_test))
print('Linear Regression Coef is : ', LinearRegressionModel.coef_)
print('Linear Regression intercept is : ', LinearRegressionModel.intercept_)
print('----------------------------------------------------')
# ----------------------------------------------------

# Calculating Prediction
y_pred = LinearRegressionModel.predict(X_test)
print('Predicted Value for Linear Regression is : ', y_pred[:5])
print('True Value for Linear Regression is : ', y_test[:5])

print('----------------------------------------------------')
# ----------------------------------------------------
# Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')  # it can be raw_values
print('Mean Absolute Error Value is : ', MAEValue)
# ----------------------------------------------------
# Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average')  # it can be raw_values
print('Mean Squared Error Value is : ', MSEValue)
# ----------------------------------------------------
# Calculating Median Squared Error
MdSEValue = median_absolute_error(y_test, y_pred)
print('Median Squared Error Value is : ', MdSEValue)
# ----------------------------------------------------
# Calculating R2 score
r2Value = r2_score(y_test, y_pred)
print('R2 score Value is : ', r2Value)
# ----------------------------------------------------
x_axis = np.arange(0,1,0.001)
x_axis = x_axis.reshape(-1,1)
print(x_axis.shape)

sns.scatterplot(x=X[:,0], y=y,alpha=0.5);
# sns.lineplot(x=x_axis[:,0], y=LinearRegressionModel.predict(x_axis));
sns.lineplot(x=X[:,0], y=LinearRegressionModel.predict(X), color='k');
plt.show()