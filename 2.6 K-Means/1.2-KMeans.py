# Import Libraries
from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import v_measure_score
# ----------------------------------------------------
'''
sklearn.cluster.KMeans(n_clusters=8, init='k-means++’, n_init=10, max_iter=300, tol=0.0001,
                       precompute_distances='auto’, verbose=0, random_state=None, copy_x=True,
                       n_jobs=None, algorithm='auto’)

=======
    - n_clusters int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    - init {‘k-means++’, ‘random’}, callable or array-like of shape (n_clusters, n_features), default=’k-means++’
        Method for initialization:
        - ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. 
            See section Notes in k_init for more details.
        - ‘random’: choose n_clusters observations (rows) at random from data for the initial centroids.
        If an array is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
        If a callable is passed, it should take arguments X, n_clusters and a random state and return an initialization.
    - n_init int, default=10
        Number of time the k-means algorithm will be run with different centroid seeds. 
        The final results will be the best output of n_init consecutive runs in terms of inertia.
    - max_iter int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
    - tol float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations
        to declare convergence.
    - precompute_distances {‘auto’, True, False}, default=’auto’
        Precompute distances (faster but takes more memory).
        ‘auto’ : do not precompute distances if n_samples * n_clusters > 12 million. 
        This corresponds to about 100MB overhead per job using double precision.
        True : always precompute distances.
        False : never precompute distances.
    - verbose int, default=0
        Verbosity mode.
    - n_jobs int, default=None
        The number of OpenMP threads to use for the computation. 
        Parallelism is sample-wise on the main cython loop which assigns each sample to its closest center.
        None or -1 means using all processors.
    - algorithm{“auto”, “full”, “elkan”}, default=”auto”
        K-means algorithm to use. The classical EM-style algorithm is “full”. 
        The “elkan” variation is more efficient on data with well-defined clusters, by using the triangle inequality. 
        However it’s more memory intensive due to the allocation of an extra array of shape (n_samples, n_clusters).
        For now “auto” (kept for backward compatibiliy) chooses “elkan” but it might change in the future for a better heuristic.
=======

=======

'''
# ----------------------------------------------------
X, y, centers = make_blobs(n_samples=1000, n_features=2, centers=3, cluster_std=2.5, center_box=(-10.0, 10.0), 
                            return_centers=True,
                            shuffle=True, 
                            random_state=17,
                            )
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
# Using the elbow method to find the optimal number of clusters
# WCSS => Within Cluster Sum of Squares

# plt.figure()
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 17)
#     kmeans.fit(X_train)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show(block=False)

# print("wcss : ", wcss)

# print("="*25)
# ----------------------------------------------------
# Applying KMeans Model
KMeansModel = KMeans(n_clusters=n_of_cluster, 
                     init='k-means++', 
                     random_state=17, 
                     )
KMeansModel.fit(X_train)
y_train_pred = KMeansModel.predict(X_train)
# ----------------------------------------------------
# Calculating Details
print('KMeansModel Score is : ', KMeansModel.score(X_train))
print('KMeansModel Score is : ', v_measure_score(y_train,y_train_pred))
print("="*25)
# ----------------------------------------------------
print('KMeansModel centers are : ', KMeansModel.cluster_centers_)
print('KMeansModel intertia is : ', KMeansModel.inertia_)
print('KMeansModel No. of iteration is : ', KMeansModel.n_iter_)
print('KMeansModel labels are : ', KMeansModel.labels_[:5])
print('KMeansModel Pred__ are : ', y_train_pred[:5])
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
print('True Value for KMeansModel is : ', y_train[:5])
# ----------------------------------------------------
x_axis = np.arange(0-0.1, 1+0.1, 0.001)
xx0, xx1 = np.meshgrid(x_axis,x_axis)
Z = KMeansModel.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

# --------

plt.figure("train")
sns.scatterplot(x=X_train[:,0], y=X_train[:,1], hue=y_train, alpha=1 , palette=['r','b','k']);
sns.scatterplot(x=centers[:,0], y=centers[:,1], s=75, color="yellow", label="Centers");
sns.scatterplot(x=KMeansModel.cluster_centers_[:,0], y=KMeansModel.cluster_centers_[:,1], s=75, color="gray", label="Centroids");
plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
plt.show(block=False) 

# --------

plt.figure("test")
sns.scatterplot(x=X_test[:,0], y=X_test[:,1], hue=y_test, alpha=1 , palette=['r','b','k']);
sns.scatterplot(x=centers[:,0], y=centers[:,1], s=75, color="yellow", label="Centers");
sns.scatterplot(x=KMeansModel.cluster_centers_[:,0], y=KMeansModel.cluster_centers_[:,1], s=75, color="gray", label="Centroids");
plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
plt.show(block=True) 

# ----------------------------------------------------
