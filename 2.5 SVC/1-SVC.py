#Import Libraries
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
# ----------------------------------------------------
'''
class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, 
                        cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', 
                        break_ties=False, random_state=None)

=======
    - C float, default=1.0
        Regularization parameter. The strength of the regularization is inversely proportional to C. 
        Must be strictly positive. The penalty is a squared l2 penalty.
    - kernel {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        Specifies the kernel type to be used in the algorithm. 
        It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. 
        If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; 
        that matrix should be an array of shape (n_samples, n_samples).
    - degree int, default=3
        Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
    - gamma {‘scale’, ‘auto’} or float, default=’scale’
        Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        - if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
        - if ‘auto’, uses 1 / n_features.
    - coef0 float, default=0.0
        Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
    - shrinking bool, default=True
        Whether to use the shrinking heuristic.
    - probability bool, default=False
        Whether to enable probability estimates. This must be enabled prior to calling fit, 
        will slow down that method as it internally uses 5-fold cross-validation, 
        and predict_proba may be inconsistent with predict.
    - tol float, default=1e-3
        Tolerance for stopping criterion.
    -cache_size float, default=200
        Specify the size of the kernel cache (in MB).
    - class_weight dict or ‘balanced’, default=None
        Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. 
        The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in 
        the input data as n_samples / (n_classes * np.bincount(y))
    - verbose bool, default=False
        Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, 
        if enabled, may not work properly in a multithreaded context.
    - max_iter int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.
    - decision_function_shape {‘ovo’, ‘ovr’}, default=’ovr’
        Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, 
        or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2).
        However, one-vs-one (‘ovo’) is always used as multi-class strategy. The parameter is ignored for binary classification.
    - break_ties bool, default=False
        If true, decision_function_shape='ovr', and number of classes > 2, 
        predict will break ties according to the confidence values of decision_function; 
        otherwise the first class among the tied classes is returned. 
        Please note that breaking ties comes at a relatively high computational cost compared to a simple predict.
    - random_state int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data for probability estimates. 
        Ignored when probability is False. 
        Pass an int for reproducible output across multiple function calls.
=======
0.0001 < gamma < 10
0.1 < c < 100

(gamma)γ ∝ 1/σ(Sigma)

kernel {linear, poly, rbf, sigmoid, precomputed}, default=rbf
=======

'''
# ----------------------------------------------------
X, y = make_classification(n_samples=1000, n_features = 2, n_informative = 2, n_redundant = 0, n_repeated = 0, 
                           n_classes = 2, n_clusters_per_class = 1, class_sep = 1.0, flip_y = 0.10, weights = [0.5,0.5], 
                           shuffle = True, random_state = 17)

print(X.shape,y.shape)
print("0 : ", len(y[y==0]))
print("1 : ",len(y[y==1]))
print("="*10)

scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------
# Applying SVC Model
SVCModel = SVC(kernel='rbf', 
               degree=3,
               C=10.0, 
               gamma='scale',
               )

SVCModel.fit(X_train, y_train)
# 0.9333333333333333
# ----------------------------------------------------
# Calculating Details
print('SVCModel Train Score is : ' , SVCModel.score(X_train, y_train))
print('SVCModel Test Score is : ' , SVCModel.score(X_test, y_test))
print("="*25)
# ----------------------------------------------------
# print("Weights assigned to the features (coefficients in the primal problem) : ", SVCModel.coef_) # coef_ is only available when using a linear kernel
# print("Coefficients of the support vector in the decision function : ", SVCModel.dual_coef_)
# print("Constants in decision function : ", SVCModel.intercept_)
# print("="*10)
# print("0 if correctly fitted, 1 otherwise (will raise warning) : ", SVCModel.fit_status_)
# print("Array dimensions of training vector X : ", SVCModel.shape_fit_)
# print("Multipliers of parameter C for each class. Computed based on the class_weight parameter. : ", SVCModel.class_weight_)
# print("="*10)
# print("Number of support vectors for each class : ", SVCModel.n_support_)
# print("Indices of support vectors : ", SVCModel.support_)
# print("Support vectors : ", SVCModel.support_vectors_)
# print("="*10)
print("The classes labels : ", SVCModel.classes_)
print("Multipliers of parameter C for each class : ", SVCModel.class_weight_)
# print("="*10)
# # If probability=True
# print("probA_ : ", SVCModel.probA_)
# print("probB_ : ", SVCModel.probB_)
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = SVCModel.predict(X_test)
print('Pred Value for SVCModel is : ' , y_pred[:5])
print('True Value for SVCModel is : ' , y_test[:5])
print("="*25)
# ----------------------------------------------------
ClassificationReport = classification_report(y_test, y_pred)
print(ClassificationReport)

print("="*10)

CM = confusion_matrix(y_test, y_pred)
print(CM)

# plt.figure()
# sns.heatmap(CM, center = True, annot=True, fmt="d")
# plt.show(block=False)

print("="*25)
# ----------------------------------------------------
x_axis = np.arange(0-0.1, 1+0.1, 0.001)
xx0, xx1 = np.meshgrid(x_axis,x_axis)
Z = SVCModel.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

plt.figure()
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1);
sns.scatterplot(x=SVCModel.support_vectors_[:,0], y=SVCModel.support_vectors_[:,1], 
                s=20, 
                color="k", 
                marker="x",
                label="support vectors"
                );
plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
plt.show(block=True) 