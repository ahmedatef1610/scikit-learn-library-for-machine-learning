# Import Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier 

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
FeatureSelection = SelectKBest(score_func=f_classif, k=2)
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
#Applying VotingClassifier Model 
#loading models for Voting Classifier
DTModel_ = DecisionTreeClassifier(criterion = 'entropy',max_depth=3,random_state = 33)
LDAModel_ = LinearDiscriminantAnalysis(n_components=1 ,solver='svd')
SGDModel_ = SGDClassifier(loss='log', penalty='l2', max_iter=10000, tol=1e-5)

#loading Voting Classifier
VotingClassifierModel = VotingClassifier(estimators=[('DTModel',DTModel_),('LDAModel',LDAModel_),('SGDModel',SGDModel_)], 
                                         voting='hard')
VotingClassifierModel.fit(X_train, y_train)
# ----------------------------------------------------
#Calculating Details
print('VotingClassifierModel Train Score is : ' , VotingClassifierModel.score(X_train, y_train))
print('VotingClassifierModel Test Score is : ' , VotingClassifierModel.score(X_test, y_test))
print("="*25)
# ----------------------------------------------------
#Calculating Prediction
y_pred = VotingClassifierModel.predict(X_test)
print('Pred Value for VotingClassifierModel is : ', y_pred[:5])
print('True Value for VotingClassifierModel is : ' , y_test[:5])
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

