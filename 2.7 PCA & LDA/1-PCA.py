# Import Libraries
from sklearn.decomposition import PCA

from sklearn.datasets import make_blobs, make_classification , load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
# Principal component analysis (PCA).
# Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. 
# The input data is centered but not scaled for each feature before applying the SVD.

'''
class sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', 
                                    random_state=None)
===
    - n_components int, float or ‘mle’, default=None
        - Number of components to keep. if n_components is not set all components are kept:
            n_components == min(n_samples, n_features)
        - If n_components == 'mle' and svd_solver == 'full', Minka’s MLE is used to guess the dimension. 
            Use of n_components == 'mle' will interpret svd_solver == 'auto' as svd_solver == 'full'.
        - If 0 < n_components < 1 and svd_solver == 'full', select the number of components such that the amount of variance that 
            needs to be explained is greater than the percentage specified by n_components.
        - If svd_solver == 'arpack', the number of components must be strictly less than the minimum of n_features and n_samples.
            Hence, the None case results in:
            n_components == min(n_samples, n_features) - 1
    - copy bool, default=True
        If False, data passed to fit are overwritten and running fit(X).transform(X) will not yield the expected results, 
        use fit_transform(X) instead.
    - whiten bool, default=False
        - When True (False by default) the components_ vectors are multiplied by the square root of n_samples and 
            then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.
        - Whitening will remove some information from the transformed signal (the relative variance scales of the components) 
            but can sometime improve the predictive accuracy of the downstream estimators by making their data respect 
            some hard-wired assumptions.
    - svd_solver {‘auto’, ‘full’, ‘arpack’, ‘randomized’}, default=’auto’
        -If auto :
            The solver is selected by a default policy based on X.shape and n_components: 
            if the input data is larger than 500x500 and the number of components to extract is lower than 80% of 
            the smallest dimension of the data, then the more efficient ‘randomized’ method is enabled. 
            Otherwise the exact full SVD is computed and optionally truncated afterwards.
        -If full :
            run exact full SVD calling the standard LAPACK solver via scipy.linalg.svd and select the components by postprocessing
        -If arpack :
            run SVD truncated to n_components calling ARPACK solver via scipy.sparse.linalg.svds. 
            It requires strictly 0 < n_components < min(X.shape)
        -If randomized :
            run randomized SVD by the method of Halko et al.
    - tol float, default=0.0
        Tolerance for singular values computed by svd_solver == ‘arpack’. Must be of range [0.0, infinity).
    - iterated_power int or ‘auto’, default=’auto’
        Number of iterations for the power method computed by svd_solver == ‘randomized’. Must be of range [0, infinity).
    - random_state int, RandomState instance or None, default=None
        Used when the ‘arpack’ or ‘randomized’ solvers are used. Pass an int for reproducible results across multiple function calls.
===

'''
# ----------------------------------------------------
print("="*25)

IrisData = load_iris()
X = IrisData.data
y = IrisData.target

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
# Applying PCAModel Model
PCAModel = PCA(n_components=2, svd_solver='auto', whiten=False, random_state=17)
X = PCAModel.fit_transform(X)
# X_train = PCAModel.fit_transform(X_train)
# X_test = PCAModel.transform(X_test)
print(X.shape)
print("="*10)
# ----------------------------------------------------
# Calculating Details
print('PCAModel Train Score is : ', PCAModel.score(X_train))
print('PCAModel Test Score is : ', PCAModel.score(X_test))
print('Compute data precision matrix with the generative model : ', PCAModel.get_precision())
# print('PCAModel Test Score Samples is : ', PCAModel.score_samples(X_test))
print("="*10)
# ----------------------------------
print('PCAModel No. of components is : ', PCAModel.components_)
print('PCAModel mean is : ', PCAModel.mean_)

print('PCAModel singular value is : ', PCAModel.singular_values_)
print('PCAModel Explained Variance is : ', PCAModel.explained_variance_)
print('PCAModel Explained Variance ratio is : ', PCAModel.explained_variance_ratio_)

print('PCAModel noise variance is : ', PCAModel.noise_variance_)

print('Number of samples in the training data is : ', PCAModel.n_samples_)
print('Number of features in the training data is : ', PCAModel.n_features_)
print('The estimated number of components is : ', PCAModel.n_components_)
print("="*25)
# ----------------------------------------------------
plt.figure("iris")
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y_hue, alpha=1, palette=['r','b','k']);
plt.show(block=True) 


