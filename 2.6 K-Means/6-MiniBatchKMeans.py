# Import Libraries
from sklearn.cluster import MiniBatchKMeans

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import v_measure_score , accuracy_score
# ----------------------------------------------------
'''
class sklearn.cluster.MiniBatchKMeans(n_clusters=8, init='k-means++', max_iter=100, batch_size=100, verbose=0, compute_labels=True, 
                                        random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, 
                                        reassignment_ratio=0.01)
===
    - batch_sizeint, default=100
        Size of the mini batches.
    - compute_labels bool, default=True
        Compute label assignment and inertia for the complete dataset once the minibatch optimization has converged in fit.
    - max_no_improvement int, default=10
        Control early stopping based on the consecutive number of mini batches 
            that does not yield an improvement on the smoothed inertia.
        To disable convergence detection based on inertia, set max_no_improvement to None.
    - init_size int, default=None
        Number of samples to randomly sample for speeding up the initialization (sometimes at the expense of accuracy): 
            the only algorithm is initialized by running a batch KMeans on a random subset of the data. 
            This needs to be larger than n_clusters.
        If None, init_size= 3 * batch_size.
    - n_init int, default=3
        Number of random initializations that are tried. 
        In contrast to KMeans, the algorithm is only run once, using the best of the n_init initializations as measured by inertia.
    - reassignment_ratio float, default=0.01
        Control the fraction of the maximum number of counts for a center to be reassigned. 
        A higher value means that low count centers are more easily reassigned, 
        which means that the model will take longer to converge, but should converge in a better clustering.
===

'''
# ----------------------------------------------------
X, y, centers = make_blobs(n_samples=1000, n_features=2, centers=None, cluster_std=2.5, center_box=(-10.0, 10.0), 
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
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
# print(X_train.shape,X_test.shape)
# print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------

# ----------------------------------------------------
# Applying MiniBatchKMeans Model
MiniBatchKMeansModel = MiniBatchKMeans(n_clusters=3, batch_size=50, init='k-means++')

MiniBatchKMeansModel.fit(X)
# kmeans = MiniBatchKMeansModel.partial_fit(X[6:12,:])
y_pred = MiniBatchKMeansModel.predict(X)
# ----------------------------------------------------
# Calculating Details
print('MiniBatchKMeansModel Score is : ', MiniBatchKMeansModel.score(X))
print('MiniBatchKMeansModel Score is : ', v_measure_score(y,y_pred))
print("="*25)
# -------------------------
print('MiniBatchKMeansModel centers are : ', MiniBatchKMeansModel.cluster_centers_)
print('MiniBatchKMeansModel intertia is : ', MiniBatchKMeansModel.inertia_)
print('MiniBatchKMeansModel No. of iteration is : ', MiniBatchKMeansModel.n_iter_)
print('MiniBatchKMeansModel labels are : ', MiniBatchKMeansModel.labels_[:5])
print('MiniBatchKMeansModel Pred__ are : ', y_pred[:5])
y_out = y_pred.copy()
y_out[y_pred==y_pred[0]] = y[0]
y_out[y_pred==y_pred[1]] = y[1]
y_out[y_pred==y_pred[4]] = y[4]
print('KMeansModel Pred__ are : ', y_out[:5])
y_pred = y_out.copy()
print('KMeansModel Score is : ', accuracy_score(y,y_pred))
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
print('True Value for KMeansModel is : ', y[:5])
print("="*25)
# ----------------------------------------------------
x_axis = np.arange(0-0.1, 1+0.1, 0.001)
xx0, xx1 = np.meshgrid(x_axis,x_axis)
Z = MiniBatchKMeansModel.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

centers = MiniBatchKMeansModel.cluster_centers_

# --------

plt.figure("data")
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1 , palette=['r','b','k']);
sns.scatterplot(x=centers[:,0], y=centers[:,1], s=75, color="yellow", label="Centers");
sns.scatterplot(x=centers[:,0], y=centers[:,1], s=75, color="gray", label="Centroids");
plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
plt.show(block=True) 
