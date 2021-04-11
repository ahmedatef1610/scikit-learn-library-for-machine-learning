#Import Libraries
from sklearn.ensemble import GradientBoostingRegressor
import sklearn.tree as sklearn_tree

from sklearn.datasets import make_blobs, make_classification, make_regression, load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#----------------------------------------------------
'''
sklearn.ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse',
                                            min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3,
                                            min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None,
                                            max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, 
                                            validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
===
    - loss {‘ls’, ‘lad’, ‘huber’, ‘quantile’}, default=’ls’
        Loss function to be optimized. ‘ls’ refers to least squares regression. 
        ‘lad’ (least absolute deviation) is a highly robust loss function solely based on order information of the input variables. 
        ‘huber’ is a combination of the two. 
        ‘quantile’ allows quantile regression (use alpha to specify the quantile).
    - subsample float, default=1.0
        The fraction of samples to be used for fitting the individual base learners. 
        If smaller than 1.0 this results in Stochastic Gradient Boosting. 
        subsample interacts with the parameter n_estimators. 
        Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
    - init estimator or ‘zero’, default=None
        An estimator object that is used to compute the initial predictions. 
        init has to provide fit and predict. 
        If ‘zero’, the initial raw predictions are set to zero. 
        By default a DummyEstimator is used, predicting either the average target value (for loss=’ls’), 
        or a quantile for the other losses.
    - alpha float, default=0.9
        The alpha-quantile of the huber loss function and the quantile loss function. Only if loss='huber' or loss='quantile'.
    - n_iter_no_change int, default=None
        n_iter_no_change is used to decide if early stopping will be used to terminate training when validation score is not improving.
        By default it is set to None to disable early stopping. If set to a number, 
        it will set aside validation_fraction size of the training data as validation and terminate training when validation score is not improving in all of the previous n_iter_no_change numbers of iterations.
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
# Applying Gradient Boosting Regressor Model 
GBRModel = GradientBoostingRegressor(   n_estimators=100, 
                                        max_depth=3,
                                        learning_rate = 0.1,
                                        loss='ls',
                                        criterion='friedman_mse',
                                        random_state=33,
                                        )
GBRModel.fit(X_train, y_train)
# ----------------------------------------------------
#Calculating Details
print('GBRModel Train Score is : ' , GBRModel.score(X_train, y_train))
print('GBRModel Test Score is : ' , GBRModel.score(X_test, y_test))
print("="*10)
# ------------------
# print("The number of classes : ",GBRModel.n_classes_)
print("The number of estimators as selected by early stopping  : ",GBRModel.n_estimators_)
# print("The collection of fitted sub-estimators : ",GBRModel.estimators_)
print()
print("The number of features when fit is performed : ",GBRModel.n_features_)
print("The inferred value of max_features : ",GBRModel.max_features_)
print("The feature importances : ",GBRModel.feature_importances_)
print()
# print("The improvement in loss (= deviance) on the out-of-bag samples relative to the previous iteration : ",GBRModel.oob_improvement_)
# print("The i-th score train_score_[i] is the deviance (= loss) of the model at iteration i on the in-bag sample : ",GBRModel.train_score_)
# print("The concrete LossFunction object : ",GBRModel.loss_)
# print("The estimator that provides the initial predictions : ",GBRModel.init_)
print()
print("="*10)
# ----------------------------------------------------
# Calculating Prediction
y_pred = GBRModel.predict(X_test)
print('Pred Value for GBRModel is : ' , y_pred[:5])
print('True Value for GBRModel is : ' , y_test[:5])
print("="*25)
# ----------------------------------------------------
x_axis = np.arange(0,1,0.001)
x_axis = x_axis.reshape(-1,1)
print(x_axis.shape)

plt.figure()
sns.scatterplot(x=X[:,0], y=y, alpha=0.5)
sns.lineplot(x=x_axis[:,0], y=GBRModel.predict(x_axis), color='k')
plt.show(block=False) 

# model = GBRModel
# plt.figure("Feature importance")
# plt.barh(range(model.n_features_), model.feature_importances_, align='center')
# plt.xlabel("Feature importance")
# plt.ylabel("Feature")
# plt.yticks(np.arange(model.n_features_), np.arange(model.n_features_)+1)
# plt.ylim(-1, model.n_features_)
# plt.show(block=True) 


plt.show(block=True) 