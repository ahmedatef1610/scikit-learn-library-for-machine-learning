#Import Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.metrics import v_measure_score , accuracy_score
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=44, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------
#Applying AggClusteringModel Model 
AggClusteringModel = AgglomerativeClustering(n_clusters=3,affinity='euclidean', linkage='ward')
# ----------------------
y_pred_train = AggClusteringModel.fit_predict(X_train)
y_pred_test = AggClusteringModel.fit_predict(X_test)
# ----------------------
print('AggClusteringModel train Score is : ', v_measure_score(y_train,y_pred_train))
print('AggClusteringModel test Score is : ', v_measure_score(y_test,y_pred_test))
# ----------------------------------------------------
# draw the Hierarchical graph for Training set
plt.figure()
dendrogram = sch.dendrogram(sch.linkage(X_train[: 30,:], method = 'ward'))
plt.title('Training Set')
plt.xlabel('X Values')
plt.ylabel('Distances')
plt.show(block=False)

# draw the Hierarchical graph for Test set
plt.figure()
dendrogram = sch.dendrogram(sch.linkage(X_test[: 30,:], method = 'ward'))
plt.title('Test Set')
plt.xlabel('X Value')
plt.ylabel('Distances')
plt.show(block=False)
# ----------------------------------------------------
# draw the Scatter for Train set
plt.figure()
plt.scatter(X_train[y_pred_train == 0, 0], X_train[y_pred_train == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(X_train[y_pred_train == 1, 0], X_train[y_pred_train == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')
plt.scatter(X_train[y_pred_train == 2, 0], X_train[y_pred_train == 2, 1], s = 10, c = 'black', label = 'Cluster 3')
plt.title('Training Set')
plt.xlabel('X Value')
plt.ylabel('y Value')
plt.legend()
plt.show(block=False)

# draw the Scatter for Test set
plt.figure()
plt.scatter(X_test[y_pred_test == 0, 0], X_test[y_pred_test == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(X_test[y_pred_test == 1, 0], X_test[y_pred_test == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')
plt.scatter(X_test[y_pred_test == 2, 0], X_test[y_pred_test == 2, 1], s = 10, c = 'black', label = 'Cluster 3')
plt.title('Testing Set')
plt.xlabel('X Value')
plt.ylabel('y Value')
plt.legend()
plt.show(block=False)

# ----------------------------------------------------
plt.show(block=True)
