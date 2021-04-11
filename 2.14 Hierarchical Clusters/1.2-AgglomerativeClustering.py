#Import Libraries
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import v_measure_score , accuracy_score
# ----------------------------------------------------
X, y, centers = make_blobs(n_samples=1000, n_features=2, centers=None, cluster_std=2.5, center_box=(-10.0, 10.0), 
                            return_centers=True,
                            shuffle=True, 
                            random_state=17,
                            )
# print(len(np.unique(y)))
n_of_cluster = len(set(y))
print(X.shape,y.shape,centers.shape,len(set(y)))
for i in range(len(set(y))):
    print(f"{i} : ", len(y[y==i]))
print("="*10)
# ---------
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
centers = scaler.transform(centers)
# ---------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
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

#draw the Hierarchical graph for Training set
plt.figure()
dendrogram = sch.dendrogram(sch.linkage(X_train[: 30,:], method = 'ward'))
plt.title('Training Set')
plt.xlabel('X Values')
plt.ylabel('Distances')
plt.show(block=False)

#draw the Hierarchical graph for Test set
plt.figure()
dendrogram = sch.dendrogram(sch.linkage(X_test[: 30,:], method = 'ward'))# it can be complete,average,single
plt.title('Test Set')
plt.xlabel('X Value')
plt.ylabel('Distances')
plt.show(block=False)
# ----------------------------------------------------

#draw the Scatter for Train set
plt.figure()
plt.scatter(X_train[y_pred_train == 0, 0], X_train[y_pred_train == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(X_train[y_pred_train == 1, 0], X_train[y_pred_train == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')
plt.scatter(X_train[y_pred_train == 2, 0], X_train[y_pred_train == 2, 1], s = 10, c = 'black', label = 'Cluster 3')
plt.title('Training Set')
plt.xlabel('X Value')
plt.ylabel('y Value')
plt.legend()
plt.show(block=False)

#draw the Scatter for Test set
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