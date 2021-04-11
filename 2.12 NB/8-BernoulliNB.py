#Import Libraries
from sklearn.naive_bayes import BernoulliNB

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
# ----------------------------------------------------
# Naive Bayes classifier for multivariate Bernoulli models. (BernoulliNB)
'''
class sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
===
    - alpha float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    - binarize float or None, default=0.0
        Threshold for binarizing (mapping to booleans) of sample features. 
        If None, input is presumed to already consist of binary vectors.
    - fit_prior bool, default=True
        Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
    - class_prior array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified the priors are not adjusted according to the data.
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
# Applying BernoulliNB Model 
BernoulliNBModel = BernoulliNB(alpha=1.0, binarize=10.0, fit_prior=True, class_prior=None)
BernoulliNBModel.fit(X_train, y_train)
# ----------------------------------------------------
# Calculating Details
print('BernoulliNBModel Train Score is : ' , BernoulliNBModel.score(X_train, y_train))
print('BernoulliNBModel Test Score is : ' , BernoulliNBModel.score(X_test, y_test))
print("="*10)
# ---------------
print('number of training samples observed in each class is : ' , BernoulliNBModel.class_count_)
print('class labels known to the classifier is : ' , BernoulliNBModel.classes_)
print('Smoothed empirical log probability for each class is : ' , BernoulliNBModel.class_log_prior_)
print("="*5)
print('Number of features of each sample is : ' , BernoulliNBModel.n_features_)
print('Number of samples encountered for each (class, feature) during fitting is : ' , BernoulliNBModel.feature_count_)
print('Empirical log probability of features given a class, P(x_i|y). is : ' , BernoulliNBModel.feature_log_prob_)
print("="*5)
print('Mirrors feature_log_prob_ for interpreting MultinomialNB as a linear model is : ' , BernoulliNBModel.coef_)
print('Mirrors class_log_prior_ for interpreting MultinomialNB as a linear model is : ' , BernoulliNBModel.intercept_)
# ---------------
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = BernoulliNBModel.predict(X_test)
y_pred_prob = BernoulliNBModel.predict_proba(X_test)
print('Prediction Probabilities Value for BernoulliNBModel is : \n', y_pred_prob[:5])
print('Pred Value for BernoulliNBModel is : ', y_pred[:5])
print('True Value for BernoulliNBModel is : ' , y_test[:5])
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
Z = BernoulliNBModel.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

plt.figure()
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1);
plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
plt.show(block=False) 


plt.show(block=True) 