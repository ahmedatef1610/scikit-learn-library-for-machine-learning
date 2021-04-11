#Import Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
FeatureSelection = SelectKBest(score_func=f_classif, k=30)
X = FeatureSelection.fit_transform(X, y)
# showing X Dimension
print('X Shape is ', X.shape)
# print(FeatureSelection.get_support())
print(BreastData.feature_names[FeatureSelection.get_support()])
print("="*25)
# -----------------------
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
# -----------------------
# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------
# Applying RandomForestClassifier Model 
RandomForestClassifierModel = RandomForestClassifier(criterion = 'gini',n_estimators=100,max_depth=None,random_state=33)
RandomForestClassifierModel.fit(X_train, y_train)
# ----------------------------------------------------
# Calculating Details
print('RandomForestClassifierModel Train Score is : ' , RandomForestClassifierModel.score(X_train, y_train))
print('RandomForestClassifierModel Test Score is : ' , RandomForestClassifierModel.score(X_test, y_test))
print("="*25)
# -----------------------
print('RandomForestClassifierModel Classes are : ', RandomForestClassifierModel.classes_)  
print('RandomForestClassifierModel The number of classes are : ', RandomForestClassifierModel.n_classes_)  
print("The number of outputs when fit is performed : ",RandomForestClassifierModel.n_outputs_)
print()
print('The number of features : ', RandomForestClassifierModel.n_features_)
print('RandomForestClassifierModel features importances are : ' , RandomForestClassifierModel.feature_importances_)
print("="*25)
# -----------------------
# Calculating Prediction
y_pred = RandomForestClassifierModel.predict(X_test)
y_pred_prob = RandomForestClassifierModel.predict_proba(X_test)
print('Prediction Probabilities Value for RandomForestClassifierModel is : \n', y_pred_prob[:5])
print('Pred Value for RandomForestClassifierModel is : ', y_pred[:5])
print('True Value for RandomForestClassifierModel is : ' , y_test[:5])
print("="*25)
# ----------------------------------------------------
ClassificationReport = classification_report(y_test, y_pred)
print(ClassificationReport)
print("="*10)
# ---------------
CM = confusion_matrix(y_test, y_pred)
print(CM)
print("="*10)
# ---------------
# plt.figure()
# sns.heatmap(CM, center = True, annot=True, fmt="d")
# plt.show(block=False)
# ---------------
print("="*25)
# ----------------------------------------------------

model = RandomForestClassifierModel
plt.figure("Feature importance")
plt.barh(range(model.n_features_), model.feature_importances_, align='center')
plt.xlabel("Feature importance")
plt.ylabel("Feature")
# plt.yticks(np.arange(model.n_features_), np.arange(model.n_features_)+1)
# plt.yticks(np.arange(model.n_features_), BreastData.feature_names)
plt.yticks(np.arange(model.n_features_), BreastData.feature_names[FeatureSelection.get_support()])
plt.ylim(-1, model.n_features_)
plt.tight_layout()
plt.show(block=False) 


plt.show(block=True) 