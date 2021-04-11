# Import Libraries
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------
# load boston data
BostonData = load_boston()
# X Data
X = BostonData.data
# y Data
y = BostonData.target
print('X Shape is ', X.shape)
print(BostonData.feature_names)
print("="*25)
# ----------------------------------------------------
# FeatureSelection = SelectKBest(score_func=f_classif, k=1)
# X = FeatureSelection.fit_transform(X, y)
# # showing X Dimension
# print('X Shape is ', X.shape)
# # print(FeatureSelection.get_support())
# print(BostonData.feature_names[FeatureSelection.get_support()])
# print("="*25)
# -----------------------
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
# -----------------------
# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------
# Applying SVR Model
SVRModel = SVR(kernel='rbf', C=20.0)
SVRModel.fit(X_train, y_train)
# ----------------------------------------------------
# Calculating Details
print('SVRModel Train Score is : ', SVRModel.score(X_train, y_train))
print('SVRModel Test Score is : ', SVRModel.score(X_test, y_test))
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = SVRModel.predict(X_test)
print('Predicted Value for SVRModel is : ', y_pred[:5])
print('True Value for SVRModel is : ', y_test[:5])
print("="*25)
# ----------------------------------------------------
# Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
print('Mean Absolute Error Value is : ', MAEValue)
# ----------------------------------------------------
# Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
print('Mean Squared Error Value is : ', MSEValue)
# ----------------------------------------------------
# Calculating Median Squared Error
MdSEValue = median_absolute_error(y_test, y_pred)
print('Median Squared Error Value is : ', MdSEValue)
# ----------------------------------------------------
# x_axis = np.arange(0,1,0.001)
# x_axis = x_axis.reshape(-1,1)
# print(x_axis.shape)

# plt.figure()
# sns.scatterplot(x=X[:,0], y=y, alpha=0.5);
# sns.lineplot(x=x_axis[:,0], y=SVRModel.predict(x_axis), color='k');
# plt.show(block=True) 