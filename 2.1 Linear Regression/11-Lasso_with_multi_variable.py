# Import Libraries
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
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
print(X.shape,y.shape)
print('X Features are ' , BostonData.feature_names)
print('----------------------------------------------------')
# ----------------------------------------------------
# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
# ----------------------------------------------------
# Applying Lasso Regression Model
LassoRegressionModel = Lasso(alpha=1.0, random_state=33, normalize=False)
LassoRegressionModel.fit(X_train, y_train)
# Calculating Details
print('Lasso Regression Train Score is : ', LassoRegressionModel.score(X_train, y_train))
print('Lasso Regression Test Score is : ', LassoRegressionModel.score(X_test, y_test))
print('Lasso Regression Coef is : ', LassoRegressionModel.coef_)
print('Lasso Regression intercept is : ', LassoRegressionModel.intercept_)
print('----------------------------------------------------')
# ----------------------------------------------------
# Calculating Prediction
y_pred = LassoRegressionModel.predict(X_test)
print('Pred Value for Lasso Regression is : ', y_pred[:5])
print('True Value for Lasso Regression is : ', y_test[:5])
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
