# Import Libraries
from sklearn.ensemble import RandomForestRegressor
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
class sklearn.ensemble.RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, 
                                                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                                                max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                                                bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, 
                                                warm_start=False, ccp_alpha=0.0, max_samples=None)

===
    -n_estimators int, default=100
        The number of trees in the forest.

    - bootstrap bool, default=True
        Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.

    - oob_score bool, default=False
        whether to use out-of-bag samples to estimate the R^2 on unseen data.
        
    - n_jobs int, default=None
        The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees. 
        None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
    
    - random_state int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) 
        and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features) 
        
    - max_samples int or float, default=None
        - If bootstrap is True, the number of samples to draw from X to train each base estimator.
        - If None (default), then draw X.shape[0] samples.
        - If int, then draw max_samples samples.
        - If float, then draw max_samples * X.shape[0] samples. Thus, max_samples should be in the interval (0, 1).
        
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
# Applying Random Forest Regressor Model 
RandomForestRegressorModel = RandomForestRegressor(n_estimators=100,
                                                   max_depth=5, 
                                                   bootstrap=True,
                                                   max_samples=None,
                                                   oob_score=True,
                                                   random_state=33,
                                                   n_jobs=-1,
                                                   )
RandomForestRegressorModel.fit(X_train, y_train)
# ----------------------------------------------------
# Calculating Details
print('Random Forest Regressor Train Score is : ' , RandomForestRegressorModel.score(X_train, y_train))
print('Random Forest Regressor Test Score is : ' , RandomForestRegressorModel.score(X_test, y_test))
print("="*10)
# ------------------
print("The number of outputs when fit is performed : ",RandomForestRegressorModel.n_outputs_)
print("The number of features when fit is performed : ",RandomForestRegressorModel.n_features_)
print('the feature importances : ' , RandomForestRegressorModel.feature_importances_)
print()
# print("The collection of fitted sub-estimators : ", RandomForestRegressorModel.estimators_)
# print("The child estimator template used to create the collection of fitted sub-estimators : ", RandomForestRegressorModel.base_estimator_)
# print()
# print("Score of the training dataset obtained using an out-of-bag estimate : ", RandomForestRegressorModel.oob_score_)
# print("Prediction computed with out-of-bag estimate on the training set : ", RandomForestRegressorModel.oob_prediction_)
# print()
print("="*10)
# ----------------------------------------------------
# Calculating Prediction
y_pred = RandomForestRegressorModel.predict(X_test)
print('Pred Value for Random Forest Regressor is : ' , y_pred[:5])
print('True Value for Random Forest Regressor is : ' , y_test[:5])
print("="*25)
# ----------------------------------------------------
x_axis = np.arange(0,1,0.001)
x_axis = x_axis.reshape(-1,1)
print(x_axis.shape)

plt.figure()
sns.scatterplot(x=X[:,0], y=y, alpha=0.5)
sns.lineplot(x=x_axis[:,0], y=RandomForestRegressorModel.predict(x_axis), color='k')
plt.show(block=False) 

# model = RandomForestRegressorModel
# plt.figure("Feature importance")
# plt.barh(range(model.n_features_), model.feature_importances_, align='center')
# plt.xlabel("Feature importance")
# plt.ylabel("Feature")
# plt.yticks(np.arange(model.n_features_), np.arange(model.n_features_)+1)
# plt.ylim(-1, model.n_features_)
# plt.show(block=True) 


plt.show(block=True) 