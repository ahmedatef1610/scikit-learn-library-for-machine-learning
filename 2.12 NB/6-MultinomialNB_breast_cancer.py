#Import Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
# ----------------------------------------------------
# load breast cancer data
BreastData = load_breast_cancer()
X = BreastData.data
y = BreastData.target

print('X Shape is ', X.shape)
print('y Shape is ', y.shape)
print(BreastData.feature_names)
print(BreastData.target_names)
print("="*10)
# ----------------------------------------------------
# 
    # chi2 => ['mean area' 'worst area']
    # GaussianNBModel Train Score is :  0.89501312335958     
    # GaussianNBModel Test Score is :  0.9361702127659575

    # f_classif => ['worst perimeter' 'worst concave points']
    # GaussianNBModel Train Score is :  0.9343832020997376
    # GaussianNBModel Test Score is :  0.9468085106382979

    # mutual_info_classif => ['worst perimeter' 'worst area']
    # GaussianNBModel Train Score is :  0.9133858267716536   
    # GaussianNBModel Test Score is :  0.9468085106382979 
# 
# FeatureSelection = SelectKBest(score_func=f_classif, k=2)
# X = FeatureSelection.fit_transform(X, y)
# # showing X Dimension
# print('X Shape after Feature Selection is ', X.shape)
# # print(FeatureSelection.get_support())
# print(BreastData.feature_names[FeatureSelection.get_support()])
# print("="*10)
# -----------------------
# scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
# X = scaler.fit_transform(X)
# -----------------------
# # Applying PCAModel Model
# PCAModel = PCA(n_components=2, random_state=17)
# X = PCAModel.fit_transform(X)
# print('X Shape after PCA is ',X.shape)
# print("="*10)
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
# Applying MultinomialNB Model 
MultinomialNBModel = MultinomialNB()
MultinomialNBModel.fit(X_train, y_train)
# ----------------------------------------------------
# Calculating Details
print('MultinomialNBModel Train Score is : ' , MultinomialNBModel.score(X_train, y_train))
print('MultinomialNBModel Test Score is : ' , MultinomialNBModel.score(X_test, y_test))
print("="*10)
# ---------------
print('number of training samples observed in each class is : ' , MultinomialNBModel.class_count_)
print('class labels known to the classifier is : ' , MultinomialNBModel.classes_)
print('Number of features of each sample is : ' , MultinomialNBModel.n_features_)
print('Number of samples encountered for each (class, feature) during fitting is : ' , MultinomialNBModel.feature_count_)
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = MultinomialNBModel.predict(X_test)
y_pred_prob = MultinomialNBModel.predict_proba(X_test)
print('Prediction Probabilities Value for MultinomialNBModel is : \n', y_pred_prob[:5])
print('Pred Value for MultinomialNBModel is : ', y_pred[:5])
print('True Value for MultinomialNBModel is : ' , y_test[:5])
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
# x_axis = np.arange(0-0.1, 1+0.1, 0.001)
# xx0, xx1 = np.meshgrid(x_axis,x_axis)
# Z = MultinomialNBModel.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

# # use feature selection
# label_arr = BreastData.feature_names[FeatureSelection.get_support()]
# plt.figure("(MultinomialNB)")
# ax = sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1);
# ax.set(xlabel=label_arr[0], ylabel=label_arr[1], title='MultinomialNB')
# plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
# plt.show(block=False) 

# use PCA
# # label_arr = BreastData.feature_names[FeatureSelection.get_support()]
# plt.figure("(MultinomialNB)")
# ax = sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1);
# # ax.set(xlabel=label_arr[0], ylabel=label_arr[1], title='GaussianNB')
# plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
# plt.show(block=False) 


plt.show(block=True) 