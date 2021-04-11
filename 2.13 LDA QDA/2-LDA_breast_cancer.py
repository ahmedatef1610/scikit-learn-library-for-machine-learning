#Import Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression
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
FeatureSelection = SelectKBest(score_func=f_classif, k=2)
X = FeatureSelection.fit_transform(X, y)
# showing X Dimension
print('X Shape after Feature Selection is ', X.shape)
# print(FeatureSelection.get_support())
print(BreastData.feature_names[FeatureSelection.get_support()])
print("="*10)
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
# Applying LDA Model 
LDAModel = LDA(n_components=1, solver='svd')
LDAModel.fit(X_train, y_train)
# ----------------------------------------------------
# Calculating Details
print('LDAModel Train Score is : ' , LDAModel.score(X_train, y_train))
print('LDAModel Test Score is : ' , LDAModel.score(X_test, y_test))
print("="*10)
# ---------------
print('LDAModel Explained Variance is : ', LDAModel.explained_variance_ratio_)
print('LDAModel classes are : ' , LDAModel.classes_)
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = LDAModel.predict(X_test)
y_pred_prob = LDAModel.predict_proba(X_test)
print('Prediction Probabilities Value for LDAModel is : \n', y_pred_prob[:5])
print('Pred Value for LDAModel is : ', y_pred[:5])
print('True Value for LDAModel is : ' , y_test[:5])
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
x_axis = np.arange(0-0.1, 1+0.1, 0.001)
xx0, xx1 = np.meshgrid(x_axis,x_axis)
Z = LDAModel.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

label_arr = BreastData.feature_names[FeatureSelection.get_support()]
plt.figure("LinearDiscriminantAnalysis (LDA)")
ax = sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1);
ax.set(xlabel=label_arr[0], ylabel=label_arr[1], title='LDA')
plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
plt.show(block=False)  

plt.show(block=True) 