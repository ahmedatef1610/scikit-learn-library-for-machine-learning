# Import Libraries
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
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
# FeatureSelection = SelectKBest(score_func=f_classif, k=5)
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
# ----------------------------------------------------
# Applying MLPRegressor Model
MLPRegressorModel = MLPRegressor(activation='relu',
                                 solver='adam',
                                 learning_rate='adaptive',
                                 hidden_layer_sizes=(100, 3),
                                 max_iter=1000,
                                 batch_size = 5,
                                 random_state=33)
MLPRegressorModel.fit(X_train, y_train)
# 0.8442390827731351
# ----------------------------------------------------
# Calculating Details
print('MLPRegressorModel Train Score is : ', MLPRegressorModel.score(X_train, y_train))
print('MLPRegressorModel Test Score is : ', MLPRegressorModel.score(X_test, y_test))
print("="*25)
# ----------------------------------------------------
print("Number of outputs : ", MLPRegressorModel.n_outputs_)
print('MLPRegressorModel last activation is : ' , MLPRegressorModel.out_activation_)
print('MLPRegressorModel No. of layers is : ' , MLPRegressorModel.n_layers_)
print('MLPRegressorModel No. of iterations is : ' , MLPRegressorModel.n_iter_)
print("The number of training samples seen by the solver during fitting : ", MLPRegressorModel.t_)
print("="*25)
# ----------------------------------------------------
print('MLPRegressorModel loss is : ' , MLPRegressorModel.loss_)
print("MLPRegressorModel best loss is : ", MLPRegressorModel.best_loss_) # early_stopping = False (must be)
print("MLPRegressorModel loss curve is : ", MLPRegressorModel.loss_curve_[-5:]) 
print("MLPRegressorModel loss curve length is : ", len(MLPRegressorModel.loss_curve_)) 
print("="*25)
plt.figure()
sns.lineplot(data=MLPRegressorModel.loss_curve_)
plt.show(block=False)
# ----------------------------------------------------
# Calculating Prediction
y_pred = MLPRegressorModel.predict(X_test)
print('Pred Value for MLPRegressorModel is : ' , y_pred[:5])
print('True Value for MLPRegressorModel is : ' , y_test[:5])
print("="*25)
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
plt.show(block=True)
