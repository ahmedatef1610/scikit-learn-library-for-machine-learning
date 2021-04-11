#Import Libraries
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import sklearn.tree as sklearn_tree

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
# ----------------------------------------------------

'''
sklearn.ensemble.VotingClassifier(estimators, voting='hard', weights=None, n_jobs=None, flatten_transform=True, verbose=False)
===
    - estimators list of (str, estimator) tuples
        Invoking the fit method on the VotingClassifier will fit clones of those original estimators 
        that will be stored in the class attribute self.estimators_. 
        An estimator can be set to 'drop' using set_params.
    - voting {‘hard’, ‘soft’}, default=’hard’
        If ‘hard’, uses predicted class labels for majority rule voting. 
        Else if ‘soft’, predicts the class label based on the argmax of the sums of the predicted probabilities, 
        which is recommended for an ensemble of well-calibrated classifiers.
    - weights array-like of shape (n_classifiers,), default=None
        Sequence of weights (float or int) to weight the occurrences of predicted class labels (hard voting) 
        or class probabilities before averaging (soft voting). Uses uniform weights if None.
    - n_jobs int, default=None
        The number of jobs to run in parallel for fit. None means 1 unless in a joblib.parallel_backend context. 
        -1 means using all processors. See Glossary for more details.
    - flatten_transform bool, default=True
        Affects shape of transform output only when voting=’soft’ If voting=’soft’ and flatten_transform=True, 
        transform method returns matrix with shape (n_samples, n_classifiers * n_classes). 
        If flatten_transform=False, it returns (n_classifiers, n_samples, n_classes).
    
===

'''

# ----------------------------------------------------
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1.0, flip_y=0.10, weights=[0.5, 0.5],
                           shuffle=True, random_state=17)

print(X.shape, y.shape)
print("0 : ", len(y[y == 0]))
print("1 : ", len(y[y == 1]))
print("="*10)
# ---------------
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
# ---------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
print("="*25)
# ----------------------------------------------------
# Applying VotingClassifier Model 
# loading models for Voting Classifier
RFModel_ = RandomForestClassifier(n_estimators=100, criterion='gini',max_depth=1, random_state=33)
LDAModel_ = LinearDiscriminantAnalysis(n_components=1 ,solver='svd')
LRModel_ = LogisticRegression()
SVCModel_ = SVC()
# loading Voting Classifier
VotingClassifierModel = VotingClassifier(estimators=[('RFModel',RFModel_),('LDAModel',LDAModel_),
                                                     ('LRModel',LRModel_),('SVCModel',SVCModel_)], voting='hard')
VotingClassifierModel.fit(X_train, y_train)
# ----------------------------------------------------
# Calculating Details
print('VotingClassifierModel Train Score is : ' , VotingClassifierModel.score(X_train, y_train))
print('VotingClassifierModel Test Score is : ' , VotingClassifierModel.score(X_test, y_test))
print("="*10)
# -----------------
# print('The collection of fitted sub-estimators : ', VotingClassifierModel.estimators_)  
# print('Attribute to access any fitted sub-estimators by name : ', VotingClassifierModel.named_estimators_)  
print("The classes labels : ",VotingClassifierModel.classes_)
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = VotingClassifierModel.predict(X_test)
# y_pred_prob = VotingClassifierModel.predict_proba(X_test)
# print('Prediction Probabilities Value for VotingClassifierModel is : ',y_pred_prob[:5])
print('Pred Value for VotingClassifierModel is : ', y_pred[:5])
print('True Value for VotingClassifierModel is : ' , y_test[:5])
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
Z = VotingClassifierModel.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

plt.figure()
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1);
plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
plt.show(block=False) 



plt.show(block=True) 