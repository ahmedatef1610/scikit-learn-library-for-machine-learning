#Import Libraries
from sklearn.neighbors import NearestNeighbors

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import v_measure_score , accuracy_score
#----------------------------------------------------
'''
sklearn.neighbors.NearestNeighbors(n_neighbors=5, radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2,
                                    metric_params=None, n_jobs=None)
===
    - radius float, default=1.0
        Range of parameter space to use by default for radius_neighbors queries.
===
'''
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
# Applying NearestNeighborsModel Model 
NearestNeighborsModel = NearestNeighbors(n_neighbors=5,
                                         radius=0.2,
                                         algorithm='auto',
                                         n_jobs=-1
                                         )
NearestNeighborsModel.fit(X_train)
#----------------------------------------------------
#Calculating Details
print('The distance metric to use is : ' , NearestNeighborsModel.effective_metric_)
print('Additional keyword arguments for the metric function is : ' , NearestNeighborsModel.effective_metric_params_)
print('Number of samples in the fitted data is : ' , NearestNeighborsModel.n_samples_fit_)
print("="*10)
print('NearestNeighborsModel Train kneighbors are : ' , NearestNeighborsModel.kneighbors(X_train[:1]))
print('NearestNeighborsModel Train radius kneighbors are : ' , NearestNeighborsModel.radius_neighbors(X_train[:1]))
print("="*10)
print('NearestNeighborsModel Test kneighbors are : ' , NearestNeighborsModel.kneighbors(X_test[:1]))
print('NearestNeighborsModel Test  radius kneighbors are : ' , NearestNeighborsModel.radius_neighbors(X_test[:1]))
print("="*25)
#----------------------------------------------------
plt.figure("data")
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1 , palette=['r','b','k']);
sns.scatterplot(x=centers[:,0], y=centers[:,1], s=75, color="yellow", label="Centers");
plt.show(block=False) 



plt.show(block=True) 