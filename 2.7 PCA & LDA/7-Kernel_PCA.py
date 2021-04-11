# Import Libraries
from sklearn.decomposition import KernelPCA

from sklearn.datasets import make_blobs, make_classification , load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
# Kernel Principal component analysis (KPCA).
# Non-linear dimensionality reduction through the use of kernels (see Pairwise metrics, Affinities and Kernels).

'''
class sklearn.decomposition.KernelPCA(n_components=None, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None, 
                                        alpha=1.0, fit_inverse_transform=False, eigen_solver='auto', tol=0, max_iter=None, 
                                        remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=None)
===
    - n_components int, default=None
        Number of components. If None, all non-zero components are kept.
    - kernel {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’}, default=’linear’
    - gamma float, default=None
        Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other kernels. 
        If gamma is None, then it is set to 1/n_features
    - degree int, default=3
        Degree for poly kernels. Ignored by other kernels.
    - coef0 float, default=1
        Independent term in poly and sigmoid kernels. Ignored by other kernels.
    - kernel_params dict, default=None
        Parameters (keyword arguments) and values for kernel passed as callable object. Ignored by other kernels.
    - alpha float, default=1.0
        Hyperparameter of the ridge regression that learns the inverse transform (when fit_inverse_transform=True).
    - fit_inverse_transform bool, default=False
        Learn the inverse transform for non-precomputed kernels. (i.e. learn to find the pre-image of a point)
    - eigen_solver  {‘auto’, ‘dense’, ‘arpack’}, default=’auto’
        Select eigen_solver to use. If n_components is much less than the number of training samples, 
        arpack may be more efficient than the dense eigen_solver.
    - tol float, default=0
        Convergence tolerance for arpack. If 0, optimal value will be chosen by arpack.
    - max_iter int, default=None
        Maximum number of iterations for arpack. If None, optimal value will be chosen by arpack.
    - remove_zero_eig bool, default=False
        If True, then all components with zero eigenvalues are removed, 
        so that the number of components in the output may be < n_components (and sometimes even zero due to numerical instability). 
        When n_components is None, this parameter is ignored and components with zero eigenvalues are removed regardless.
    - n_jobs int, default=None
        The number of parallel jobs to run. None means 1 unless in a joblib.parallel_backend context. 
        -1 means using all processors. See Glossary for more details.
===

'''
# ----------------------------------------------------
print("="*25)

IrisData = load_iris()
X = IrisData.data
y = IrisData.target
#------------------------
y_hue = pd.Series(y)
y_hue[y == 0] = IrisData.target_names[0]
y_hue[y == 1] = IrisData.target_names[1]
y_hue[y == 2] = IrisData.target_names[2]
print(X.shape, IrisData.feature_names)
print(y.shape, IrisData.target_names)
# print(y)
# print(y_hue.values)
print("="*10)
# ---------
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
# ---------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=16, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ---------
# Applying KernelPCA Model
KPCAModel = KernelPCA(n_components = 2, kernel="rbf")
X = KPCAModel.fit_transform(X,y)
# X_train = KPCAModel.fit_transform(X_train)
# X_test = KPCAModel.transform(X_test)
print(X.shape)
print("="*10)
# ----------------------------------------------------
# Calculating Details
# print('KPCAModel Eigenvalues of the centered kernel matrix in decreasing order: ', KPCAModel.lambdas_)
# print('KPCAModel Inverse transform matrix  is : ', KPCAModel.dual_coef_) # fit_inverse_transform is True
# print('KPCAModel Eigenvectors of the centered kernel matrix is : ', KPCAModel.alphas_)

# print('KPCAModel Projection of the fitted data on the kernel principal components is : ', KPCAModel.X_transformed_fit_)
# print('KPCAModel The data used to fit the model is : ', KPCAModel.X_fit_)

print("="*25)
# ----------------------------------------------------
plt.figure("iris")
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y_hue, alpha=1, palette=['r','b','k']);
plt.show(block=True) 

# plt.figure("iris train")
# sns.scatterplot(x=X_train[:,0], y=X_train[:,1], hue=y_hue, alpha=1, palette=['r','b','k']);
# plt.show(block=False) 
# plt.figure("iris test")
# sns.scatterplot(x=X_test[:,0], y=X_test[:,1], hue=y_hue, alpha=1, palette=['r','b','k']);
# plt.show(block=True) 

