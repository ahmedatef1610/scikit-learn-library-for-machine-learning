#Import Libraries
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import sklearn.tree as sklearn_tree

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
FeatureSelection = SelectKBest(score_func=f_classif, k=13)
X = FeatureSelection.fit_transform(X, y)
# showing X Dimension
print('X Shape is ', X.shape)
# print(FeatureSelection.get_support())
print(BostonData.feature_names[FeatureSelection.get_support()])
print("="*25)
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
# Applying Random Forest Regressor Model 
RandomForestRegressorModel = RandomForestRegressor(n_estimators=100,max_depth=None, random_state=33)
RandomForestRegressorModel.fit(X_train, y_train)
# ----------------------------------------------------
# Calculating Details
print('Random Forest Regressor Train Score is : ' , RandomForestRegressorModel.score(X_train, y_train))
print('Random Forest Regressor Test Score is : ' , RandomForestRegressorModel.score(X_test, y_test))
print("="*25)
# ----------------------------------------------------
print("The number of outputs when fit is performed : ",RandomForestRegressorModel.n_outputs_)
print()
print("The number of features when fit is performed : ",RandomForestRegressorModel.n_features_)
print("the feature importances : ",RandomForestRegressorModel.feature_importances_)
print()
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = RandomForestRegressorModel.predict(X_test)
print('Pred Value for Random Forest Regressor is : ' , y_pred[:5])
print('True Value for DecisionTreeRegressorModel is : ', y_test[:5])
print("="*25)
#----------------------------------------------------
# Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
print('Mean Absolute Error Value is : ', MAEValue)
#--------------------
#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
print('Mean Squared Error Value is : ', MSEValue)
#--------------------
#Calculating Median Squared Error
MdSEValue = median_absolute_error(y_test, y_pred)
print('Median Squared Error Value is : ', MdSEValue )
#--------------------
print("="*25)
#----------------------------------------------------


model = RandomForestRegressorModel
plt.figure("Feature importance")
plt.barh(range(model.n_features_), model.feature_importances_, align='center')
plt.xlabel("Feature importance")
plt.ylabel("Feature")
# plt.yticks(np.arange(model.n_features_), BostonData.feature_names)
plt.yticks(np.arange(model.n_features_), BostonData.feature_names[FeatureSelection.get_support()])
# plt.ylim(-1, model.n_features_)
plt.tight_layout()
plt.show(block=True) 


