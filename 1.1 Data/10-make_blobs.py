# Import Libraries
from sklearn.datasets import make_blobs

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------

# load clustering data

'''
sklearn.datasets.make_blobs(n_samples=100, n_features=2, centers=None, cluster_std=1.0, center_box=(-10.0, 10.0), 
                                shuffle=True, random_state=None, return_centers=False)
=======
centers=None => centers=3
=======
'''
# ----------------------------------------------------

X, y, centers = make_blobs(n_samples=1000, n_features=2, centers=3, cluster_std=2.5, center_box=(-10.0, 10.0), 
                            return_centers=True,
                            shuffle=True, 
                            random_state=17,
                            )

# X Data
# print('X Data is \n', X[:5])
print('X shape is ', X.shape)

# y Data
print('y Data is \n', y[:10])
print('y shape is ', y.shape)

# centers Data
print('centers Data is \n', centers[:10])
print('centers shape is ', centers.shape)

# --------------
plt.figure()
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1);
sns.scatterplot(x=centers[:,0], y=centers[:,1], 
                s=100, color="yellow", label="Centers"
                );
plt.show(block=True) 
# # --------------
# scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
# X = scaler.fit_transform(X)
# # --------------
# plt.figure()
# sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1);
# plt.show(block=True) 
# ----------------------------------------------------

# X, y = make_blobs(n_samples=10, centers=3, n_features=2, random_state=0)
# print(X.shape)
# print(y)

# print("="*10)

# X, y = make_blobs(n_samples=[3, 3, 10], centers=None, n_features=2, random_state=0)
# print(X.shape)
# print(y)
