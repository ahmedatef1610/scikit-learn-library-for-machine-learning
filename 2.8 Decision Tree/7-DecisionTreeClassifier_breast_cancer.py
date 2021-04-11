#Import Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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
# Applying DecisionTreeClassifier Model 
DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='gini',
                                                     max_depth=5,
                                                     random_state=33)
DecisionTreeClassifierModel.fit(X_train, y_train)
#----------------------------------------------------
# Calculating Details
print('DecisionTreeClassifierModel Train Score is : ' , DecisionTreeClassifierModel.score(X_train, y_train))
print('DecisionTreeClassifierModel Test Score is : ' , DecisionTreeClassifierModel.score(X_test, y_test))
print("="*10)
#--------------------------
print('DecisionTreeClassifierModel Classes are : ', DecisionTreeClassifierModel.classes_)  
print('DecisionTreeClassifierModel The number of classes are : ', DecisionTreeClassifierModel.n_classes_)  
print("The number of outputs when fit is performed : ",DecisionTreeClassifierModel.n_outputs_)
print()
print("The number of features when fit is performed : ",DecisionTreeClassifierModel.n_features_)
print("The inferred value of max_features : ",DecisionTreeClassifierModel.max_features_)
print("the feature importances : ",DecisionTreeClassifierModel.feature_importances_)
print()
print("="*10)
#--------------------------
# Calculating Prediction
y_pred = DecisionTreeClassifierModel.predict(X_test)
y_pred_prob = DecisionTreeClassifierModel.predict_proba(X_test)
print('Prediction Probabilities Value for DecisionTreeClassifierModel is : \n', y_pred_prob[:5])
print('Pred Value for DecisionTreeClassifierModel is : ', y_pred[:5])
print('True Value for SVCModel is : ' , y_test[:5])
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
# Z = DecisionTreeClassifierModel.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

# plt.figure()
# sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1);
# plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
# plt.show(block=False) 

# plt.figure('Decision Tree')
# sklearn_tree.plot_tree(DecisionTreeClassifierModel)
# plt.show(block=False)


model = DecisionTreeClassifierModel
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