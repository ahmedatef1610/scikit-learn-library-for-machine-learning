# Import Libraries
from sklearn.model_selection import cross_validate

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
sklearn.model_selection.cross_validate(estimator, X, y=None, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, 
                                        fit_params=None, pre_dispatch='2*n_jobs', return_train_score=False, 
                                        return_estimator=False, error_score=nan)

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
# Applying Cross Validate :
#  don't forget to define the model first !!!
# CrossValidateValues1 = cross_validate(LinearRegressionModel, X, y, cv=4, return_train_score=True)
# CrossValidateValues2 = cross_validate(LinearRegressionModel, X, y, cv=4, return_train_score=True, scoring=('r2', 'neg_mean_squared_error'))
CrossValidateValues = cross_validate(LinearRegressionModel, X, y, 
                                      cv=4, 
                                      return_train_score=True,
                                      scoring=('r2', 'neg_mean_squared_error'), 
                                      n_jobs=-1)
# ----------------------------------------------------
# Showing Results

# print('- Train Score Value : ', CrossValidateValues1['train_score'])
# print('- Test Score Value : ', CrossValidateValues1['test_score'])
# print('- Fit Time : ', CrossValidateValues1['fit_time'])
# print('- Score Time : ', CrossValidateValues1['score_time'])

# # print('- Train MSE Value : ', CrossValidateValues2['train_neg_mean_squared_error'])
# print('- Test MSE Value : ', CrossValidateValues2['test_neg_mean_squared_error'])
# # print('- Train R2 Value : ', CrossValidateValues2['train_r2'])
# print('- Test R2 Value : ', CrossValidateValues2['test_r2'])
# --------------
print(CrossValidateValues)
print("="*10)

# print('- Train Score Value : ', CrossValidateValues['train_score'])
# print('- Test Score Value : ', CrossValidateValues['test_score'])
print('- Fit Time : ', CrossValidateValues['fit_time'])
print('- Score Time : ', CrossValidateValues['score_time'])

print('- Train MSE Value : ', CrossValidateValues['train_neg_mean_squared_error'])
print('- Test MSE Value : ', CrossValidateValues['test_neg_mean_squared_error'])
print('- Train R2 Value : ', CrossValidateValues['train_r2'])
print('- Test R2 Value : ', CrossValidateValues['test_r2'])
print("="*10)
for key in CrossValidateValues.keys():
    print('value of ' , key , ' is  ' , CrossValidateValues[key])
# ----------------------------------------------------
