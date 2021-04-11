from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
iris = load_iris()
X = iris.data
y = iris.target

print(X.shape, iris.feature_names)
print(y.shape, iris.target_names)
print("="*25)
# ----------------------------------------------------
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
# ----------------
result = kmeans.labels_
print(silhouette_score(X, result))
print("="*25)
# ----------------------------------------------------
score = []
for n in range(2, 11):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X)
    result = kmeans.labels_
    print(n, '    ', silhouette_score(X, result))
    print("="*10)
    score.append(kmeans.inertia_)

plt.figure()
plt.plot(range(2, 11), score)
plt.show(block=False)
# ----------------------------------------------------
kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit_predict(X)
# ----------------------------------------------------
plt.figure()
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=10, c='r')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=10, c='b')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=10, c='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='y')
plt.show(block=True)
# ----------------------------------------------------
