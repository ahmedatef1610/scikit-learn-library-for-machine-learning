import numpy as np
import pandas as pd
# ----------------------------------------------------
dataset = pd.read_csv('path/2.9 Ensemble Reg/houses.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# ----------------------------------------------------
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X)
X= imp.transform(X)
# ----------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)
# ----------------------------------------------------
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X_train, y_train)
# ----------------------------------------------------
print(regressor.score(X_train, y_train))
print(regressor.score(X_test, y_test))
print("="*25)
print(regressor.feature_importances_)
print("="*25)
# ----------------------------------------------------
# Predicting a new result
y_pred = regressor.predict(X_test)
print(y_pred[:5])
print(y_test[:5])
print("="*25)



