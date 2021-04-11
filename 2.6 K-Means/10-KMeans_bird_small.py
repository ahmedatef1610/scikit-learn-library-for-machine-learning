# Import Libraries
from sklearn.cluster import KMeans

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
# ----------------------------------------------------
# we need to compress the image 
image_data = loadmat('path/2.6 K-Means/bird_small.mat')
# image_data
print(image_data.keys())
# ---------------
# image
A = image_data['A']
# print(A) # image
print(A.shape)
plt.figure("image")
plt.imshow(A)
plt.show(block=False)
# ---------------
# normalize value ranges
A = A / 255.
# reshape the array
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
print(X.shape)
print("="*25)
# ----------------------------------------------------
# Applying KMeans Model
KMeansModel = KMeans(n_clusters=2, init='k-means++', random_state=17)
KMeansModel.fit(X)
print('KMeansModel centers are : ', KMeansModel.cluster_centers_)
# print('KMeansModel labels are : ', KMeansModel.labels_[:5])
print("="*25)
# ----------------------------------------------------
label = KMeansModel.predict(X)
print(label.shape)
print(label[:])
print("="*25)
# ---------------
# map each pixel to the centroid value
X_recovered = KMeansModel.cluster_centers_[label.astype(int),:]
print(X_recovered.shape)
print(X_recovered[:5])
print("="*25)
# ---------------
# reshape to the original dimensions
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))
plt.figure("image after clustersing")
plt.imshow(X_recovered)
plt.show(block=False)
# ----------------------------------------------------
plt.imsave("path/2.6 K-Means/image before clustersing.png", A)
plt.imsave("path/2.6 K-Means/image after clustersing.png", X_recovered)
# ----------------------------------------------------


plt.show(block=True)



