# Import Libraries
from sklearn.svm import SVR

from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
'''
class sklearn.svm.SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, 
                        cache_size=200, verbose=False, max_iter=- 1)

=======
    - kernel {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        Specifies the kernel type to be used in the algorithm. 
        It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. 
        If none is given, ‘rbf’ will be used. 
        If a callable is given it is used to precompute the kernel matrix.
    - degree int, default=3
        Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
    - gamma {‘scale’, ‘auto’} or float, default=’scale’
        Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        - if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
        - if ‘auto’, uses 1 / n_features. 
    - oef0 float, default=0.0
        Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’
    - tol float, default=1e-3
        Tolerance for stopping criterion.
    - C float, default=1.0
        Regularization parameter. The strength of the regularization is inversely proportional to C. 
        Must be strictly positive. The penalty is a squared l2 penalty.
    - epsilon float, default=0.1
        Epsilon in the epsilon-SVR model. 
        It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted 
        within a distance epsilon from the actual value.
    - shrinking bool, default=True
        Whether to use the shrinking heuristic. 
    - cache_size float, default=200
        Specify the size of the kernel cache (in MB).
    - verbose bool, default=False
        Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, 
        if enabled, may not work properly in a multithreaded context.
    - max_iter int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.
=======
0.0001 < gamma < 10
0.1 < c < 100

(gamma)γ ∝ 1/σ(Sigma)

kernel {linear, poly, rbf, sigmoid, precomputed}, default=rbf
=======

'''
# ----------------------------------------------------
X, y = make_regression(n_samples=10000, n_features=1, shuffle=True, noise=25, random_state=16)
print(X.shape, y.shape)
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------
# Applying SVR Model
SVRModel = SVR(kernel='rbf', 
               degree=3,
               C=5.0, 
               epsilon=0.1,
               gamma='scale',
               )
SVRModel.fit(X_train, y_train)
# ----------------------------------------------------
# Calculating Details
print('SVRModel Train Score is : ', SVRModel.score(X_train, y_train))
print('SVRModel Test Score is : ', SVRModel.score(X_test, y_test))
print("="*25)
# ------------------
# print("Weights assigned to the features (coefficients in the primal problem) : ", SVRModel.coef_) # coef_ is only available when using a linear kernel
print("Coefficients of the support vector in the decision function : ", SVRModel.dual_coef_)
print("Constants in decision function : ", SVRModel.intercept_)
print("="*10)
print("0 if correctly fitted, 1 otherwise (will raise warning) : ", SVRModel.fit_status_)
print("Array dimensions of training vector X : ", SVRModel.shape_fit_)
print("Multipliers of parameter C for each class. Computed based on the class_weight parameter. : ", SVRModel.class_weight_)
print("="*10)
print("Number of support vectors for each class : ", SVRModel.n_support_)
print("Indices of support vectors : ", SVRModel.support_)
print("Support vectors : ", SVRModel.support_vectors_)
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = SVRModel.predict(X_test)
print('Predicted Value for SVRModel is : ', y_pred[:5])
print('True Value for SVRModel is : ', y_test[:5])
print("="*25)
# ----------------------------------------------------
x_axis = np.arange(0,1,0.001)
x_axis = x_axis.reshape(-1,1)
print(x_axis.shape)

# print("="*50)
# print(y[SVRModel.support_])
# print("="*50)

plt.figure()
sns.scatterplot(x=X[:,0], y=y, alpha=0.5);
sns.lineplot(x=x_axis[:,0], y=SVRModel.predict(x_axis), color='k');
# sns.scatterplot(x=SVRModel.support_vectors_[:,0], y=y[SVRModel.support_], 
#                 s=5, 
#                 color="k", 
#                 marker="x",
#                 label="support vectors"
#                 );
plt.show(block=True) 