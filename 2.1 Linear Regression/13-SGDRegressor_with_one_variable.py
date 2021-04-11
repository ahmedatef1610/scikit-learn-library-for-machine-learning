# Import Libraries
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error

from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
'''
class sklearn.linear_model.SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, 
                                        fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, 
                                        epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, 
                                        power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, 
                                        warm_start=False, average=False)
---
loss str, default=’squared_loss’
The loss function to be used. 
The possible values are ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’
---
penalty {‘l2’, ‘l1’, ‘elasticnet’}, default=’l2’
---
alpha float, default=0.0001
---
l1_ratio float, default=0.15
The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. 
l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Only used if penalty is ‘elasticnet’.
---
fit_intercept bool, default=True
---
max_iter int, default=1000
---
tol float, default=1e-3
The stopping criterion. 
If it is not None, training will stop when (loss > best_loss - tol) for n_iter_no_change consecutive epochs.
---
shuffle bool, default=True
Whether or not the training data should be shuffled after each epoch.
---
verbose int, default=0
The verbosity level.
---
epsilon float, default=0.1
Epsilon in the epsilon-insensitive loss functions; only if loss is 
‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’. 
For ‘huber’, determines the threshold at which it becomes less important to get the prediction exactly right. 
For epsilon-insensitive, any differences between the current prediction and the correct label are ignored 
if they are less than this threshold.
---

---
'''
# ----------------------------------------------------
X, y = make_regression(n_samples=10000, n_features=1, shuffle=True, noise=25, random_state=16)
print(X.shape,y.shape)
print("="*25)
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
# ----------------------------------------------------
# Applying SGDRegressor Model
SGDRegressionModel = SGDRegressor(random_state=33)
SGDRegressionModel.fit(X_train, y_train)

# Calculating Details
print('SGD Regression Train Score is : ', SGDRegressionModel.score(X_train, y_train))
print('SGD Regression Test Score is : ', SGDRegressionModel.score(X_test, y_test))
print('SGD Regression Coef is : ', SGDRegressionModel.coef_)
print('SGD Regression intercept is : ', SGDRegressionModel.intercept_)
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = SGDRegressionModel.predict(X_test)
print('Pred Value for SGD Regression is : ', y_pred[:5])
print('True Value for SGD Regression is : ', y_test[:5])
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
x_axis = np.arange(0,1,0.001)
x_axis = x_axis.reshape(-1,1)
print(x_axis.shape)

sns.scatterplot(x=X[:,0], y=y,alpha=0.5);
# sns.lineplot(x=x_axis[:,0], y=LinearRegressionModel.predict(x_axis));
sns.lineplot(x=X[:,0], y=SGDRegressionModel.predict(X));
plt.show()