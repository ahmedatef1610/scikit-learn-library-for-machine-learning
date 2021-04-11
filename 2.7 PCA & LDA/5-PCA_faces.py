import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn.decomposition import PCA
# ----------------------------------------------------
faces =  loadmat('path/2.7 PCA/ex7faces.mat')
print(faces.keys())
image_num = 1610
# -----------------------
X = faces['X']
print(X.shape) # 5000 image => (32,32) == 1024
# -----------------------
plt.figure("images before PCA")
plt.imshow(X)
plt.show(block=False)
# -----------------------
# show one face
face = np.reshape(X[image_num,:], (32, 32))
plt.figure("image before PCA")
plt.imshow(face)
plt.show(block=False)
# ----------------------------------------------------
PCAModel = PCA(n_components=100, svd_solver='auto')
X = PCAModel.fit_transform(X)
# ----------------------------------------------------
print(X.shape)
# -----------------------
plt.figure('images after PCA')
plt.imshow(X)
plt.show(block=False)
# -----------------------
# show one face
face = np.reshape(X[image_num,:], (10, 10))
plt.figure("image after PCA")
plt.imshow(face)
plt.show(block=False)
# ----------------------------------------------------
X = PCAModel.inverse_transform(X)
# ----------------------------------------------------
print(X.shape)
# -----------------------
plt.figure('images recover from PCA')
plt.imshow(X)
plt.show(block=False)
# -----------------------
# show one face
face = np.reshape(X[image_num,:], (32, 32))
plt.figure('image recover from PCA')
plt.imshow(face)
plt.show(block=True)
# ----------------------------------------------------
