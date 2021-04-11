#Import Libraries
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
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
FeatureSelection = SelectKBest(score_func=f_classif, k=5)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=44, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------
# Applying KNeighborsRegressor Model 
KNeighborsRegressorModel = KNeighborsRegressor(n_neighbors = 5, 
                                               weights='uniform', 
                                               algorithm = 'auto',
                                               p = 2,
                                               n_jobs=-1,
                                               )
KNeighborsRegressorModel.fit(X_train, y_train)
# ----------------------------------------------------
#Calculating Details
print('KNeighborsRegressorModel Train Score is : ' , KNeighborsRegressorModel.score(X_train, y_train))
print('KNeighborsRegressorModel Test Score is : ' , KNeighborsRegressorModel.score(X_test, y_test))
print("="*10)
# ----------------------
print('The distance metric to use is : ' , KNeighborsRegressorModel.effective_metric_)
print('Additional keyword arguments for the metric function is : ' , KNeighborsRegressorModel.effective_metric_params_)
print('Number of samples in the fitted data is : ' , KNeighborsRegressorModel.n_samples_fit_)
print("="*25)
# ----------------------------------------------------
y_pred = KNeighborsRegressorModel.predict(X_test)
print('Pred Value for KNeighborsRegressorModel is : ' , y_pred[:5])
print('True Value for KNeighborsRegressorModel is : ', y_test[:5])
print("="*25)
# ----------------------------------------------------
#Calculating Mean Absolute Error
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
# ------------------------
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 21 to good selection num neighbors
neighbors_settings = range(1, 40)
for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsRegressor(n_neighbors=n_neighbors,weights='uniform',algorithm = 'auto',p = 2,n_jobs=-1,)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
plt.figure('choose', figsize=(10,5))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.xticks(np.arange(0,40,1))
plt.legend()
plt.tight_layout()
plt.show(block=False)

# ------------------------
plt.show(block=True) 