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

# from joblib import Memory
# cachedir = 'path/2.14 Hierarchical Clusters/cache'
# mem = Memory(cachedir)
# ----------------------------------------------------
# Agglomerative Clustering

# Recursively merges the pair of clusters that minimally increases a given linkage distance.
'''
sklearn.cluster.AgglomerativeClustering(n_clusters=2, affinity='euclidean', memory=None, connectivity=None, 
                                            compute_full_tree='auto', linkage='ward', distance_threshold=None, 
                                            compute_distances=False)

=======
    - n_clusters int or None, default=2
        The number of clusters to find. It must be None if distance_threshold is not None.
    - affinity str or callable, default=’euclidean’
        Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”.
        If linkage is “ward”, only “euclidean” is accepted. 
        If “precomputed”, a distance matrix (instead of a similarity matrix) is needed as input for the fit method.
    - memory str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree. 
        By default, no caching is done. If a string is given, it is the path to the caching directory.
    - connectivity array-like or callable, default=None
        Connectivity matrix. Defines for each sample the neighboring samples following a given structure of the data. 
        This can be a connectivity matrix itself or a callable that transforms the data into a connectivity matrix, 
        such as derived from kneighbors_graph. Default is None, i.e, the hierarchical clustering algorithm is unstructured.
    - compute_full_tree ‘auto’ or bool, default=’auto’
        Stop early the construction of the tree at n_clusters. 
        This is useful to decrease computation time if the number of clusters is not small compared to the number of samples. 
        This option is useful only when specifying a connectivity matrix.
    - linkage {‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
        Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. 
        The algorithm will merge the pairs of cluster that minimize this criterion.
        ‘ward’ minimizes the variance of the clusters being merged.
        ‘average’ uses the average of the distances of each observation of the two sets.
        ‘complete’ or ‘maximum’ linkage uses the maximum distances between all observations of the two sets.
        ‘single’ uses the minimum of the distances between all observations of the two sets.
    - distance_threshold float, default=None
        The linkage distance threshold above which, clusters will not be merged. 
        If not None, n_clusters must be None and compute_full_tree must be True.
    - compute_distances bool, default=False
        Computes distances between clusters even if distance_threshold is not used. 
        This can be used to make dendrogram visualization, but introduces a computational and memory overhead.
=======

'''
# ----------------------------------------------------
X, y, centers = make_blobs(n_samples=1000, n_features=2, centers=3, cluster_std=2.5, center_box=(-10.0, 10.0), 
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
# Applying AggClusteringModel Model 
AggClusteringModel = AgglomerativeClustering(n_clusters=n_of_cluster, 
                                             affinity='euclidean', 
                                             linkage='ward',
                                             compute_distances= True,
                                             )
# y_pred_train = AggClusteringModel.fit_predict(X_train)
# y_pred_test = AggClusteringModel.fit_predict(X_test)
y_pred = AggClusteringModel.fit_predict(X)
# ----------------------
print('AggClusteringModel Score is : ', v_measure_score(y,y_pred))
y_out = y_pred.copy()
y_out[y_pred==y_pred[0]] = y[0]
y_out[y_pred==y_pred[1]] = y[1]
y_out[y_pred==y_pred[4]] = y[4]
y_pred = y_out.copy()
print('AggClusteringModel Score is : ', accuracy_score(y,y_pred))
print("="*25)
# ----------------------------------------------------
print('The number of clusters is : ', AggClusteringModel.n_clusters_)
print('Number of leaves is : ', AggClusteringModel.n_leaves_)
print('The estimated number of connected components in the graph is : ', AggClusteringModel.n_connected_components_)
print('The children of each non-leaf node is : ', AggClusteringModel.children_[:5])
# print('Distances between nodes is : ', AggClusteringModel.distances_[:5]) #compute_distances is set to True
print("="*25)
# ---------------------
print('cluster labels for each point is : ', AggClusteringModel.labels_[:8])
print('Pred Value for AggClusteringModel is : ', y_pred[:8])
print('True Value for AggClusteringModel is : ', y[:8])
print("="*25)
# ----------------------------------------------------
ClassificationReport = classification_report(y, y_pred)
print(ClassificationReport)
print("="*10)
# ---------------
CM = confusion_matrix(y, y_pred)
print(CM)
print("="*10)
# ---------------
plt.figure()
sns.heatmap(CM, center = True, annot=True, fmt="d")
plt.show(block=False)
# ---------------
print("="*25)
# ----------------------------------------------------
# draw the Hierarchical graph for Training set
# dendrogram = sch.dendrogram(sch.linkage(X[: 30,:], method = 'ward'))
# plt.title('Training Set')
# plt.xlabel('X Values')
# plt.ylabel('Distances')
# plt.show()

# # draw the Hierarchical graph for Test set
# dendrogram = sch.dendrogram(sch.linkage(X_test[: 30,:], method = 'ward'))
# plt.title('Test Set')
# plt.xlabel('X Value')
# plt.ylabel('Distances')
# plt.show()

# import scipy.cluster.hierarchy as sch
plt.figure("Dendrogram for Dataset")
dendrogram = sch.dendrogram(sch.linkage(X[:50,:], method = 'ward'))
plt.title('Dataset')
plt.xlabel('X Values')
plt.ylabel('Distances')
plt.show(block=False)
# ----------------------------------------------------
# x_axis = np.arange(0-0.1, 1+0.1, 0.001)
# xx0, xx1 = np.meshgrid(x_axis,x_axis)
# Z = AggClusteringModel.fit_predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)
# ---------------
y_label = [ ("Cluster 0" if x==0 else ("Cluster 1" if x==1 else ("Cluster 2" if x==2 else x))) for x in y]
y_label_pred = [ ("Cluster 0" if x==0 else ("Cluster 1" if x==1 else ("Cluster 2" if x==2 else x))) for x in y_pred]

plt.figure("data")
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y_label, alpha=1 , palette=['r','b','k']);
sns.scatterplot(x=centers[:,0], y=centers[:,1], s=75, color="yellow", label="Centers");
# plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
plt.show(block=False) 


plt.figure("data in end")
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y_label_pred, alpha=1 , palette=['r','b','k']);
sns.scatterplot(x=centers[:,0], y=centers[:,1], s=75, color="yellow", label="Centers");
# plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
plt.show(block=False) 


plt.show(block=True) 
