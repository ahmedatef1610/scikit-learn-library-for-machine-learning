#Import Libraries
from sklearn.cluster import DBSCAN

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import v_measure_score , accuracy_score
# ----------------------------------------------------
# Perform DBSCAN clustering from vector array or distance matrix.
# DBSCAN - Density-Based Spatial Clustering of Applications with Noise. 
# Finds core samples of high density and expands clusters from them. 
# Good for data which contains clusters of similar density.

'''
sklearn.cluster.DBSCAN(eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, 
                        p=None, n_jobs=None)

=======
    - eps float, default=0.5
        The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
        This is not a maximum bound on the distances of points within a cluster. 
        This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
    - min_samples int, default=5
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        This includes the point itself.
    - metric string, or callable, default=’euclidean’
        The metric to use when calculating distance between instances in a feature array. 
        If metric is a string or callable, it must be one of the options allowed by sklearn.metrics.pairwise_distances for 
        its metric parameter. 
        If metric is “precomputed”, X is assumed to be a distance matrix and must be square. 
        X may be a Glossary, in which case only “nonzero” elements may be considered neighbors for DBSCAN.
    - metric_params dict, default=None
        Additional keyword arguments for the metric function.
    - algorithm {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
        The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors. 
    - leaf_size int, default=30
        Leaf size passed to BallTree or cKDTree. 
        This can affect the speed of the construction and query, as well as the memory required to store the tree. 
        The optimal value depends on the nature of the problem.
    - p float, default=None
        The power of the Minkowski metric to be used to calculate distance between points. 
        If None, then p=2 (equivalent to the Euclidean distance).
    - n_jobs int, default=None
        The number of parallel jobs to run. None means 1 unless in a joblib.parallel_backend context. 
        -1 means using all processors. See Glossary for more details.


=======
'''
# ----------------------------------------------------
X, y, centers = make_blobs(n_samples=1000, n_features=2, centers=None, cluster_std=1.0, center_box=(-10.0, 10.0), 
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
# scaler = StandardScaler()
X = scaler.fit_transform(X)
centers = scaler.transform(centers)
# ---------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------
# Applying DBSCANModel Model 
DBSCANModel = DBSCAN(   eps=0.1,
                        metric='euclidean',
                        min_samples=5,
                        algorithm='auto',
                        )

# y_pred_train = DBSCANModel.fit_predict(X_train)
# y_pred_test = DBSCANModel.fit_predict(X_test)
y_pred = DBSCANModel.fit_predict(X)
# ----------------------------------------------------
# Calculating Details
print('DBSCANModel Score is : ', v_measure_score(y,y_pred))
y_out = y_pred.copy()
y_out[y_pred==y_pred[0]] = y[0]
y_out[y_pred==y_pred[1]] = y[1]
y_out[y_pred==y_pred[4]] = y[4]
y_pred = y_out.copy()
print('AggClusteringModel Score is : ', accuracy_score(y,y_pred))
print("="*5)
# ---------------------
# print('Indices of core samples : ' ,DBSCANModel.core_sample_indices_)
# print('Copy of each core sample found by training : ' ,DBSCANModel.components_)
# print('DBSCANModel labels are : ' ,DBSCANModel.labels_[:8])
# print("="*5)
# ---------------------
# print('DBSCANModel Train data are : ' ,y_pred_train)
# print('DBSCANModel Test data are : ' ,y_pred_test)
print('Pred Value for DBSCANModel is : ', y_pred[:8])
print('True Value for DBSCANModel is : ', y[:8])
print("="*25)
# ----------------------------------------------------
x_axis = np.arange(0-0.1, 1+0.1, 0.01)
xx0, xx1 = np.meshgrid(x_axis,x_axis)
# Z = DBSCANModel.fit_predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

# ----------------------

plt.figure("data")
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1 , palette=['r','b','k']);
sns.scatterplot(x=centers[:,0], y=centers[:,1], s=75, color="yellow", label="Centers");
# plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
plt.show(block=False) 


plt.figure("data in end")
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y_pred, alpha=1, palette=['r','b','k'] );
sns.scatterplot(x=centers[:,0], y=centers[:,1], s=75, color="yellow", label="Centers");
# plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
plt.show(block=False) 


plt.show(block=True) 