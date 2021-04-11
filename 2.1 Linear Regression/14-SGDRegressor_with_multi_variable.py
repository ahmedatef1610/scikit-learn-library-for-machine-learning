# Import Libraries
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
# ----------------------------------------------------
# load boston data
BostonData = load_boston()
# X Data
X = BostonData.data
# y Data
y = BostonData.target
# ----------------------------------------------------
# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
# ----------------------------------------------------
# Applying SGDRegressor Model
SGDRegressionModel = SGDRegressor(random_state=33)
SGDRegressionModel.fit(X_train, y_train) 

# Calculating Details
print('SGD Regression Train Score is : ',SGDRegressionModel.score(X_train, y_train))
print('SGD Regression Test Score is : ',SGDRegressionModel.score(X_test, y_test))
print('SGD Regression Coef is : ', SGDRegressionModel.coef_)
print('SGD Regression intercept is : ', SGDRegressionModel.intercept_)
print('-'*25)
# ----------------------------------------------------

# Calculating Prediction
y_pred = SGDRegressionModel.predict(X_test)
print('Pred Value for SGD Regression is : ', y_pred[:5])
print('True Value for SGD Regression is : ', y_test[:5])

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
