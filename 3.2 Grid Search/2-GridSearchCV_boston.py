#Import Libraries
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
#----------------------------------------------------
#load boston data
BostonData = load_boston()
X = BostonData.data
y = BostonData.target
#----------------------------------------------------
#Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)
#----------------------------------------------------
#Applying Grid Searching :  
#Example : 
from sklearn.svm import SVR
SelectedModel = SVR(epsilon=0.1,gamma='auto')
SelectedParameters = {'kernel':('linear', 'rbf'), 'C':[1,2,3,4,5]}
#---------------------
GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters, cv = 2, return_train_score=True)
GridSearchModel.fit(X_train, y_train)
#---------------------

sorted(GridSearchModel.cv_results_.keys())
GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]
#---------------------

# Showing Results
print('All Results are :\n', GridSearchResults )
print("="*25)

print('Best Score is :', GridSearchModel.best_score_)
print('Best Parameters are :', GridSearchModel.best_params_)
print('Best Estimator is :', GridSearchModel.best_estimator_)
print("="*25)
