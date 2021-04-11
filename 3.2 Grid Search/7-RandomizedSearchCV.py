#Import Libraries
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import pandas as pd

from sklearn.svm import SVR

from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#----------------------------------------------------
'''                         
model_selection.RandomizedSearchCV(estimator, param_distributions, n_iter=10, scoring=None, n_jobs=None,
                                                    refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, 
                                                    error_score=nan, return_train_score=False)


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

parameters = [{'kernel':['linear','rbf'], 'C':[1,2,3,4,5]},
              {'kernel':['poly'], 'C':[1,2,3,4,5],'degree':[1,2,3,4,5,6]},
              ]

parameters = [{'kernel':['rbf'], 'C':[1]},
              {'kernel':['poly'], 'C':[5],'degree':[1,2,3]},
              ]
#----------------------------------------------------
RandomizedSearchModel = RandomizedSearchCV(SelectedModel, parameters, cv=2, return_train_score=True, n_jobs = -1)
RandomizedSearchModel.fit(X_train, y_train)
#---------------------
# sorted(RandomizedSearchModel.cv_results_.keys())
RandomizedSearchResults = pd.DataFrame(RandomizedSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]
#---------------------

# Showing Results
print('All Results are :\n', RandomizedSearchResults )
print('Best Score is :', RandomizedSearchModel.best_score_)
print('Best Parameters are :', RandomizedSearchModel.best_params_)
print('Best Estimator is :', RandomizedSearchModel.best_estimator_)