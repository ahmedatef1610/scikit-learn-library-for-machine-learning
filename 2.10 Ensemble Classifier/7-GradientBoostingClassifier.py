#Import Libraries
from sklearn.ensemble import GradientBoostingClassifier
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
class sklearn.ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                                                    criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                                    min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, 
                                                    min_impurity_split=None, init=None, random_state=None, max_features=None, 
                                                    verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1,
                                                    n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
===
    - loss{‘deviance’, ‘exponential’}, default=’deviance’
        The loss function to be optimized. ‘deviance’ refers to deviance (= logistic regression) 
        for classification with probabilistic outputs. For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm.
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
GBCModel = GradientBoostingClassifier(n_estimators=100,max_depth=3,random_state=33) 
GBCModel.fit(X_train, y_train)
# ----------------------------------------------------
#Calculating Details
print('GBCModel Train Score is : ' , GBCModel.score(X_train, y_train))
print('GBCModel Test Score is : ' , GBCModel.score(X_test, y_test))
print("="*10)
# --------------------
print('The number of estimators : ', GBCModel.n_estimators_)  
print('GBCModel Classes are : ', GBCModel.classes_)  
print('GBCModel The number of classes are : ', GBCModel.n_classes_)  
print("The number of data features: ",GBCModel.n_features_)
print()
print('The inferred value of max_features : ' , GBCModel.max_features)
print('GBCModel features importances are : ' , GBCModel.feature_importances_)
print("="*25)
# ----------------------------------------------------
#Calculating Prediction
y_pred = GBCModel.predict(X_test)
y_pred_prob = GBCModel.predict_proba(X_test)
print('Prediction Probabilities Value for GBCModel is : ' , y_pred_prob[:5])
print('Pred Value for GBCModel is : ' , y_pred[:5])
print('True Value for GBCModel is : ' , y_test[:5])
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
Z = GBCModel.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

plt.figure()
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1);
plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
plt.show(block=False) 



model = GBCModel
plt.figure("Feature importance")
plt.barh(range(model.n_features_), model.feature_importances_, align='center')
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.yticks(np.arange(model.n_features_), np.arange(model.n_features_)+1)
plt.ylim(-1, model.n_features_)
plt.show(block=False) 


plt.show(block=True) 