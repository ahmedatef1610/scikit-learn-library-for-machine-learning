from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
# ----------------------------------------------------
# generating two clusters: x with 10 points and y with 10:
np.random.seed(1234)
x = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[10,])
y = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[10,])
print(x)
print("="*5)
print(y)
print("="*25)
# ----------------
X = np.concatenate((x, y),)
print(X)
print(X.shape)
print("="*25)
# ----------------
plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.show(block=False)
# ----------------------------------------------------
# generate the linkage matrix
Z = linkage(X, 'ward')
# print(Z)
# print("="*25)
# -------------
# coph_dists = cophenet(Z, pdist(X))
# print(coph_dists)
# print("="*25)
# -------------
plt.figure(figsize=(10, 5))
dendrogram(Z, leaf_rotation=90, leaf_font_size=12)
plt.title('HCA Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.show(block=False)
# ----------------------------------------------------
plt.show(block=True)


