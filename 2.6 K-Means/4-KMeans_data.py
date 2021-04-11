from sklearn.cluster import KMeans

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
dataset = pd.read_csv('path/2.6 K-Means/data.csv')
X = dataset.iloc[:,:].values

print(X.shape)
# ----------------------------------------------------
wcss = []
n = 20
for i in range(1,n):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.figure()
plt.plot(range(1,n), wcss)
plt.title('Elbow')
plt.xlabel('clusters')
plt.ylabel('inertias')
plt.show(block=False)
# ----------------------------------------------------
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)
# ----------------------------------------------------
# # Visualising the clusters
# plt.figure()
# plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 10, c = 'r')
# plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 10, c = 'b')
# plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 10, c = 'g')
# plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 10, c = 'c')
# plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 10, c = 'm')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'y')
# plt.title('Clusters of customers')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.legend()
# plt.show(block=False) 
# ----------------------------------------------------
# x_axis = np.arange(0-0.1, 1+0.1, 0.001)
# xx0, xx1 = np.meshgrid(x_axis,x_axis)
# Z = kmeans.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

palette = sns.color_palette("Set2", 6)
plt.figure()
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y_kmeans, alpha=1, palette=plt.cm.tab20);
sns.scatterplot(x=kmeans.cluster_centers_[:,0], y=kmeans.cluster_centers_[:,1], s=100, color="y", label="centers");
# plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.tab20)
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show(block=True) 