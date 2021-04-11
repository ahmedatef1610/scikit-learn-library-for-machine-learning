#Import Libraries
from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
# ----------------------------------------------------
# Gaussian Naive Bayes (GaussianNB)
'''
class sklearn.naive_bayes.GaussianNB(priors=None, var_smoothing=1e-09)
===
    - priors array-like of shape (n_classes,)
        Prior probabilities of the classes. If specified the priors are not adjusted according to the data.
    - var_smoothing float, default=1e-9
        Portion of the largest variance of all features that is added to variances for calculation stability.
===

'''

# ----------------------------------------------------
X, y = make_classification(n_samples=1000, n_features = 2, n_informative = 2, n_redundant = 0, n_repeated = 0, 
                           n_classes = 2, n_clusters_per_class = 1, class_sep = 1.0, flip_y = 0.10, weights = [0.5,0.5], 
                           shuffle = True, random_state = 17)

print(X.shape,y.shape)
print("0 : ", len(y[y==0]))
print("1 : ",len(y[y==1]))
print("="*10)
# ---------------
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
# ---------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------
#Applying GaussianNB Model 
GaussianNBModel = GaussianNB()
GaussianNBModel.fit(X_train, y_train)
# GaussianNBModel Train Score is :  0.8791044776119403
# GaussianNBModel Test Score is :  0.8787878787878788
# ==========
# ----------------------------------------------------
# Calculating Details
print('GaussianNBModel Train Score is : ' , GaussianNBModel.score(X_train, y_train))
print('GaussianNBModel Test Score is : ' , GaussianNBModel.score(X_test, y_test))
print("="*10)
# ---------------
print('probability of each class is : ' , GaussianNBModel.class_prior_)
print('number of training samples observed in each class is : ' , GaussianNBModel.class_count_)
print('class labels known to the classifier is : ' , GaussianNBModel.classes_)
print("="*5)
print('absolute additive value to variances is : ' , GaussianNBModel.epsilon_)
print('variance of each feature per class is : ' , GaussianNBModel.sigma_)
print('mean of each feature per class is : ' , GaussianNBModel.theta_)

print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = GaussianNBModel.predict(X_test)
y_pred_prob = GaussianNBModel.predict_proba(X_test)
print('Prediction Probabilities Value for GaussianNBModel is : \n', y_pred_prob[:5])
print('Pred Value for GaussianNBModel is : ', y_pred[:5])
print('True Value for GaussianNBModel is : ' , y_test[:5])
print("="*25)
# ----------------------------------------------------
ClassificationReport = classification_report(y_test, y_pred)
print(ClassificationReport)
print("="*10)
# ---------------
CM = confusion_matrix(y_test, y_pred)
print(CM)
print("="*10)
# ---------------
# plt.figure()
# sns.heatmap(CM, center = True, annot=True, fmt="d")
# plt.show(block=False)
# ---------------
print("="*25)
# ----------------------------------------------------
x_axis = np.arange(0-0.1, 1+0.1, 0.001)
xx0, xx1 = np.meshgrid(x_axis,x_axis)
Z = GaussianNBModel.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

plt.figure()
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1);
plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
plt.show(block=False) 


plt.show(block=True) 