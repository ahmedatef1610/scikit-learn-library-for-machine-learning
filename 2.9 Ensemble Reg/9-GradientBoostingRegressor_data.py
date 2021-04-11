import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  GradientBoostingRegressor
# ----------------------------------------------------
features = pd.read_csv('path/2.9 Ensemble Reg/data.csv')
print(features.head())
print("="*10)
print(features.describe())
print("="*10)
print('The shape of our features is:', features.shape)
print("="*25)
# ----------------------
features = pd.get_dummies(features)
# features.iloc[:,5:].head(5)
print(features.head())
print('The shape of our features is:', features.shape)
print("="*25)
# ----------------------
labels = np.array(features['actual'])
features = features.drop('actual', axis = 1)
feature_list = list(features.columns)
features = np.array(features)
print(feature_list)
# ----------------------
print("="*25)
# ----------------------------------------------------
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
print("="*25)
# ----------------------------------------------------
model = GradientBoostingRegressor(n_estimators = 100 , learning_rate = 0.1, max_depth = 3)
model.fit(train_features, train_labels)
# ----------------------------------------------------
print('Gradient Boosting Regressor Train Score is : ' , model.score(train_features, train_labels))
print('Gradient Boosting Regressor Test Score is : ' , model.score(test_features, test_labels))
print("="*10)
# ----------------------------------------------------
predictions = model.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# ------------------------
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
# ----------------------------------------------------



