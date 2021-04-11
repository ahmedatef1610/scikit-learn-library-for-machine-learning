# Import Libraries
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
'''
class sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
                                                intercept_scaling=1, class_weight=None, random_state=None, 
                                                solver='lbfgs', max_iter=100, multi_class='auto', 
                                                verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
---
solver {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
Algorithm to use in the optimization problem.
For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; 
‘liblinear’ is limited to one-versus-rest schemes.
‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty
‘liblinear’ and ‘saga’ also handle L1 penalty
‘saga’ also supports ‘elasticnet’ penalty
‘liblinear’ does not support setting penalty='none'
---
'''
# ----------------------------------------------------
X, y = make_classification(n_samples=1000, n_features = 2, n_informative = 2, n_redundant = 0, n_repeated = 0, 
                           n_classes = 2, n_clusters_per_class = 1, class_sep = 1.0, flip_y = 0.10, weights = [0.5,0.5], 
                           shuffle = True, random_state = 17)

print(X.shape,y.shape)
print(len(y[y==0]))
print(len(y[y==1]))
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
# ----------------------------------------------------
# Applying LogisticRegression Model
LogisticRegressionModel = LogisticRegression(penalty='l2', solver='sag', C=1.0, n_jobs=-1, random_state=33)
LogisticRegressionModel.fit(X_train, y_train)

# Calculating Details
print('LogisticRegressionModel Train Score is : ', LogisticRegressionModel.score(X_train, y_train))
print('LogisticRegressionModel Test Score is : ', LogisticRegressionModel.score(X_test, y_test))
print('LogisticRegressionModel Classes are : ', LogisticRegressionModel.classes_)
print('LogisticRegressionModel No. of iterations is : ', LogisticRegressionModel.n_iter_)
print('-'*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = LogisticRegressionModel.predict(X_test)
y_pred_prob = LogisticRegressionModel.predict_proba(X_test)
print('Prediction Probabilities Value for LogisticRegressionModel is : ', y_pred_prob[:5])
print('Predicted Value for LogisticRegressionModel is : ', y_pred[:5])
print('True Value for LogisticRegressionModel is : ', y_test[:5])
# ----------------------------------------------------
# x_axis = np.arange(0,1,0.001)
x_axis = np.arange(0-0.1, 1+0.1, 0.001)

# x0_min, x0_max = x_axis.min()-0.1, x_axis.max()+0.1
# x1_min, x1_max = x_axis.min()-0.1, x_axis.max()+0.1
# xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, 0.001), np.arange(x1_min, x1_max, 0.001))
xx0, xx1 = np.meshgrid(x_axis,x_axis)
Z = LogisticRegressionModel.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1);
plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
# sns.lineplot(x=x_axis[:,0], y=LinearRegressionModel.predict(x_axis));
plt.show()

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, y_pred)
sns.heatmap(CM, center = True, annot=True, fmt="d")
plt.show()
# ----------------------------


def plotData(X, y ,size):
    
    pos = (y == 1)
    neg = (y == 0)
    
    plt.scatter(X[pos, 0], X[pos, 1], s = size, marker='d', c= 'b', linewidths=1, label = 1)
    plt.scatter(X[neg, 0], X[neg, 1], s = size, marker='x', c= 'r', linewidths=1, label = 0)


def plot_data(clf, X, y, h=0.003, pad=0.25, size=100):
    
    x0_min, x0_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    x1_min, x1_max = X[:, 1].min()-pad, X[:, 1].max()+pad

    xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h))

    # Z = clf.predict(np.c_[xx0.ravel(), xx1.ravel()])
    Z = clf.predict(np.array([xx0.ravel(), xx1.ravel()]).T).reshape(xx0.shape)

    plt.contourf(xx0, xx1, Z, cmap=plt.cm.Paired, alpha=0.2)
    plotData(X, y, size)

    plt.xlim(x0_min, x0_max)
    plt.ylim(x1_min, x1_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()


# plotData(X, y, 10)
# plt.show()
# plot_data(LogisticRegressionModel, X, y, h=0.003, pad=0.25, size=20)
# ----------------------------------------------------
