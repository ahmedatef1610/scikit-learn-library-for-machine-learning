#Import Libraries
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MinMaxScaler

#----------------------------------------------------
#load boston data
BostonData = load_boston()
#X Data
X = BostonData.data
#y Data
y = BostonData.target
#----------------------------------------------------
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
# -----------------------
#Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)
#----------------------------------------------------
#Applying SVRModel Model 
SVRModel = SVR(kernel='rbf', C=20.0)
SVRModel.fit(X_train, y_train)
#----------------------------------------------------
# Applying Cross Validate Score :  
# don't forget to define the model first !!!
CrossValidateScoreTrain = cross_val_score(SVRModel, X_train, y_train, cv=3)
CrossValidateScoreTest = cross_val_score(SVRModel, X_test, y_test, cv=3)
CrossValidateScore = cross_val_score(SVRModel, X, y, cv=3)

# Showing Results
print('Cross Validate Score for Training Set: ', CrossValidateScoreTrain)
print('Cross Validate Score for Testing  Set: ', CrossValidateScoreTest)
print('Cross Validate Score for data     Set: ', CrossValidateScore)
# ------------------------

accuracies = cross_val_score(SVRModel, X, y, cv=20)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))