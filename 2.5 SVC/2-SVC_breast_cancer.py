# Import Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
# ----------------------------------------------------
# load breast cancer data
BreastData = load_breast_cancer()
# X Data
X = BreastData.data
# y Data
y = BreastData.target

print('X Shape is ', X.shape)
print(BreastData.feature_names)
print(BreastData.target_names)
print("="*25)
# ----------------------------------------------------
# FeatureSelection = SelectKBest(score_func=f_classif, k=1)
# X = FeatureSelection.fit_transform(X, y)
# # showing X Dimension
# print('X Shape is ', X.shape)
# # print(FeatureSelection.get_support())
# print(BostonData.feature_names[FeatureSelection.get_support()])
# print("="*25)
# -----------------------
# scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
# X = scaler.fit_transform(X)
# -----------------------
# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------
# Applying SVC Model
SVCModel = SVC(kernel='rbf',
               degree=3,
               C=10.0, 
               gamma='scale',
               )
SVCModel.fit(X_train, y_train)
# ----------------------------------------------------
# Calculating Details
print('SVCModel Train Score is : ', SVCModel.score(X_train, y_train))
print('SVCModel Test Score is : ', SVCModel.score(X_test, y_test))
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = SVCModel.predict(X_test)
print('Pred Value for SVCModel is : ' , y_pred[:5])
print('True Value for SVCModel is : ' , y_test[:5])
print("="*25)
# ----------------------------------------------------
# Calculating Confusion Matrix - classification_report
ClassificationReport = classification_report(y_test, y_pred)
print(ClassificationReport)

print("="*10)

CM = confusion_matrix(y_test, y_pred)
print(CM)

plt.figure()
sns.heatmap(CM, center = True, annot=True, fmt="d")
plt.show(block=False)

print("="*25)
# ----------------------------------------------------
plt.show(block=True)
