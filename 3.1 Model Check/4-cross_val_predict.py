#Import Libraries
from sklearn.model_selection import cross_val_predict

from sklearn.datasets import load_boston, make_regression

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import MinMaxScaler
# ----------------------------------------------------
'''
sklearn.model_selection.cross_val_predict(estimator, X, y=None, groups=None, cv=None, n_jobs=None, verbose=0, 
                                            fit_params=None, pre_dispatch='2*n_jobs', method='predict')
===

===

'''
# ----------------------------------------------------
X, y = make_regression(n_samples=10000, n_features=1, shuffle=True, noise=25, random_state=16)
print(X.shape, y.shape)
print("="*10)
# ---------------
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
# ---------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=44, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------
# Applying Linear Regression Model
LinearRegressionModel = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=-1)
LinearRegressionModel.fit(X_train, y_train)
# ----------------------------------------------------
# Calculating Details
print('LinearRegressionModel Train Score is : ', LinearRegressionModel.score(X_train, y_train))
print('LinearRegressionModel Test Score is : ', LinearRegressionModel.score(X_test, y_test))
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = LinearRegressionModel.predict(X_test)
print('Pred Value for Linear Regression is : ', y_pred[:5])
print('True Value for Linear Regression is : ', y_test[:5])
print("="*25)
# ----------------------------------------------------
x_axis = np.arange(0,1,0.001)
x_axis = x_axis.reshape(-1,1)
print(x_axis.shape)
print("="*50)
# ------------------
plt.figure()
sns.scatterplot(x=X[:,0], y=y, alpha=0.5)
sns.lineplot(x=x_axis[:,0], y=LinearRegressionModel.predict(x_axis), color='k')
plt.show(block=True) 
# ----------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# Applying Cross Validate Predict :
# don't forget to define the model first !!!

# CrossValidatePredictionTrain = cross_val_predict(LinearRegressionModel, X_train, y_train, cv=3)
# CrossValidatePredictionTest = cross_val_predict(LinearRegressionModel, X_test, y_test, cv=3)
CrossValidatePrediction = cross_val_predict(LinearRegressionModel, X, y, cv=3)

# Showing Results
# print('Cross Validate Prediction for Training Set: \n', CrossValidatePredictionTrain[:10])
# print('Cross Validate Prediction for Testing Set: \n', CrossValidatePredictionTest[:10])
print('Cross Validate Prediction for dataset: \n', CrossValidatePrediction[:5])