#Import Libraries
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_blobs, make_classification, make_regression, load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
'''
sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', 
                                        metric_params=None, n_jobs=None, **kwargs)
===
    - n_neighbors int, default=5
        Number of neighbors to use by default for kneighbors queries.
    - weights{‘uniform’, ‘distance’} or callable, default=’uniform’
        weight function used in prediction. Possible values:
        - ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
        - ‘distance’ : weight points by the inverse of their distance. 
            in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an array of distances, 
            and returns an array of the same shape containing the weights.
    - algorithm {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
        Algorithm used to compute the nearest neighbors:
        ‘ball_tree’ will use BallTree
        ‘kd_tree’ will use KDTree
        ‘brute’ will use a brute-force search.
        ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.
        Note: fitting on sparse input will override the setting of this parameter, using brute force.
    - leaf_size int, default=30
        Leaf size passed to BallTree or KDTree. 
        This can affect the speed of the construction and query, as well as the memory required to store the tree. 
        The optimal value depends on the nature of the problem.
    - p int, default=2
        Power parameter for the Minkowski metric. 
        When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) 
        for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    - metric str or callable, default=’minkowski’
        the distance metric to use for the tree. The default metric is minkowski, 
        and with p=2 is equivalent to the standard Euclidean metric. 
        See the documentation of DistanceMetric for a list of available metrics. 
        If metric is “precomputed”, X is assumed to be a distance matrix and must be square during fit. 
        X may be a sparse graph, in which case only “nonzero” elements may be considered neighbors.
    - metric_params dict, default=None
        Additional keyword arguments for the metric function.
    
===
weights {'uniform', 'distance'} or callable, default=’uniform’
algorithm {'auto', 'ball_tree', 'kd_tree', 'brute'}, default=’auto’
p { 1 ,2 } default=2
===

'''
# ----------------------------------------------------
X, y = make_regression(n_samples=10000, n_features=1, shuffle=True, noise=25, random_state=16)
print(X.shape, y.shape)
print("="*10)
# ---------------
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
# ---------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=44, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------
# Applying KNeighborsRegressor Model 
KNeighborsRegressorModel = KNeighborsRegressor(n_neighbors = 30, 
                                               weights='uniform', 
                                               algorithm = 'auto',
                                               p = 2,
                                               n_jobs=-1,
                                               ) 
KNeighborsRegressorModel.fit(X_train, y_train)
# ----------------------------------------------------
# Calculating Details
print('KNeighborsRegressorModel Train Score is : ' , KNeighborsRegressorModel.score(X_train, y_train))
print('KNeighborsRegressorModel Test Score is : ' , KNeighborsRegressorModel.score(X_test, y_test))
print("="*10)
# ----------------------
print('The distance metric to use is : ' , KNeighborsRegressorModel.effective_metric_)
print('Additional keyword arguments for the metric function is : ' , KNeighborsRegressorModel.effective_metric_params_)
print('Number of samples in the fitted data is : ' , KNeighborsRegressorModel.n_samples_fit_)

print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = KNeighborsRegressorModel.predict(X_test)
print('Pred Value for KNeighborsRegressorModel is : ', y_pred[:5])
print('True Value for KNeighborsRegressorModel is : ', y_test[:5])
print("="*25)
# ----------------------------------------------------
x_axis = np.arange(0,1,0.001)
x_axis = x_axis.reshape(-1,1)
print(x_axis.shape)

plt.figure('KNN')
sns.scatterplot(x=X[:,0], y=y, alpha=0.5)
sns.lineplot(x=x_axis[:,0], y=KNeighborsRegressorModel.predict(x_axis), color='k')
plt.show(block=False) 
# ------------------------
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 21 to good selection num neighbors
neighbors_settings = range(1, 40)
for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsRegressor(n_neighbors=n_neighbors,weights='uniform',algorithm = 'auto',p = 2,n_jobs=-1,)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
plt.figure('choose', figsize=(10,7))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.xticks(np.arange(0,40,1))
plt.legend()
plt.tight_layout()
plt.show(block=False)

# ------------------------
plt.show(block=True) 