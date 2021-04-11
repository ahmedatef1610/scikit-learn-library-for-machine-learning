# Import Libraries
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
'''
class sklearn.linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, 
                                        max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, 
                                        random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, 
                                        validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)


---

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
print('-'*25)
# ----------------------------------------------------
# Applying SGDClassifier Model
SGDClassifierModel = SGDClassifier(penalty='l2', loss='squared_loss', learning_rate='optimal', random_state=33)
SGDClassifierModel.fit(X_train, y_train)
# Calculating Details
print('SGDClassifierModel Train Score is : ', SGDClassifierModel.score(X_train, y_train))
print('SGDClassifierModel Test Score is : ', SGDClassifierModel.score(X_test, y_test))
print('SGDClassifierModel loss function is : ', SGDClassifierModel.loss_function_)
print('SGDClassifierModel Classes are : ', SGDClassifierModel.classes_)
print('SGDClassifierModel No. of iterations is : ', SGDClassifierModel.n_iter_)
print('-'*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = SGDClassifierModel.predict(X_test)
print('Predicted Value for SGDClassifierModel is : ', y_pred[:5])
print('True Value for SGDClassifierModel is : ', y_test[:5])
print('-'*25)
# ----------------------------------------------------
x_axis = np.arange(0-0.1, 1+0.1, 0.001)
xx0, xx1 = np.meshgrid(x_axis,x_axis)
Z = SGDClassifierModel.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1);
plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
plt.show()

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, y_pred)
sns.heatmap(CM, center = True, annot=True, fmt="d")
plt.show()
# ----------------------------------------------------
