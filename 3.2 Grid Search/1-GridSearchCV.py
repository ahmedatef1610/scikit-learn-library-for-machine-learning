#Import Libraries
from sklearn.model_selection import GridSearchCV
import pandas as pd

from sklearn.svm import SVR

from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#----------------------------------------------------

#Applying Grid Searching :  
'''
model_selection.GridSearchCV(estimator, param_grid, scoring=None,fit_params=None, n_jobs=None, iid=’warn’,
                             refit=True, cv=’warn’, verbose=0,pre_dispatch=‘2*n_jobs’, error_score=
                             ’raisedeprecating’,return_train_score=’warn’)

===

===
'''
#---------------------
'''
    class sklearn.svm.SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, 
                            cache_size=200, verbose=False, max_iter=- 1)

    0.0001 < gamma < 10
    0.1 < c < 100
    (gamma)γ ∝ 1/σ(Sigma)
    kernel {linear, poly, rbf, sigmoid, precomputed}, default=rbf

    'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
=======

'''
# ----------------------------------------------------
X, y = make_regression(n_samples=10000, n_features=1, shuffle=True, noise=25, random_state=16)
print(X.shape, y.shape)
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------
# Applying SVR Model
# SVRModel = SVR(kernel='rbf', 
#                degree=3,
#                C=5.0,
#                epsilon=0.1,
#                gamma='scale',
#                )

SVRModel = SVR()
SelectedModel = SVRModel
SelectedParameters = {'kernel':['linear','poly','rbf','sigmoid'], 
                        'C':[0.5,1,2,3,4,5,10,100,1000],
                        'degree':[1,2,3,4,5,6],
                        'gamma':['scale','auto',0.1,0.2,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10],
                    }
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
              ]

parameters = {'kernel':['linear','poly','rbf','sigmoid'],
              'gamma':['scale','auto',0.1,0.5,1,2,5,10],
              }
parameters = {'kernel':['linear','poly','rbf','sigmoid']}

#-----------------------------------------------------------------------------------------
GridSearchModel = GridSearchCV(estimator=SelectedModel, param_grid=parameters, cv=2, return_train_score=True, n_jobs = -1)
GridSearchModel.fit(X_train, y_train)
#---------------------
sorted(GridSearchModel.cv_results_.keys())
GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]
#---------------------

# Showing Results
print('All Results are :\n', GridSearchResults )
print("="*25)
print('Best Estimator is :', GridSearchModel.best_estimator_)
print('Best Score is :', GridSearchModel.best_score_)
print('Best Parameters are :', GridSearchModel.best_params_)
print('Best index Parameters are :', GridSearchModel.best_index_)
print("="*25)
print('Scorer function used on the held out data to choose the best parameters for the model :', GridSearchModel.scorer_)
print('The number of cross-validation splits are :', GridSearchModel.n_splits_)
print('Seconds used for refitting the best model on the whole dataset are :', GridSearchModel.refit_time_)
print('Whether or not the scorers compute several metricst are :', GridSearchModel.multimetric_)
print("="*25)
print(GridSearchModel.cv_results_.keys())
print("="*10)
GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)
print(GridSearchResults)
# GridSearchResults.to_csv("path/3.2 Grid Search/g.csv",index=False)