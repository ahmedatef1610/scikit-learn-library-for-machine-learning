from sklearn.cluster import KMeans

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
trainingdata = np.random.rand(200, 2)
X = np.array(trainingdata)
print(trainingdata)
print("="*25)
# ----------------------------------------------------
testdata = np.random.rand(20, 2)
print(testdata)
print("="*25)
# ----------------------------------------------------
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)
print(kmeans.labels_)
print(kmeans.inertia_)
print("="*25)
# ----------------------------------------------------
kmeans.predict(np.array(testdata))
centers = kmeans.cluster_centers_
print(centers)
print("="*25)
# ----------------------------------------------------
# plt.figure()
# plt.scatter(trainingdata[:, 0], trainingdata[:, 1], c='r')
# plt.scatter(testdata[:, 0], testdata[:, 1], c='b')

# for j in range(len(centers)):
#     plt.scatter(centers[j, 0], centers[j, 1], c='y', s=100)

# plt.show(block=False)
# ----------------------------------------------------
x_axis = np.arange(0-0.1, 1+0.1, 0.001)
xx0, xx1 = np.meshgrid(x_axis,x_axis)
Z = kmeans.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

plt.figure()
sns.scatterplot(x=trainingdata[:,0], y=trainingdata[:,1], color='r', label="training data");
sns.scatterplot(x=testdata[:,0], y=testdata[:,1], color='b', label="test data");
sns.scatterplot(x=centers[:,0], y=centers[:,1], s=100, color="y", label="centers");
plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.tab20)
plt.show(block=True) 
