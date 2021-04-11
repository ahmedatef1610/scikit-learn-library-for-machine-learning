# Import Libraries
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as sklearn_tree

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
# ----------------------------------------------------
# A decision tree classifier.

'''
class sklearn.tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                                            min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
                                            min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, ccp_alpha=0.0)
===
    - criterion {“gini”, “entropy”}, default=”gini”
        The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for 
        the information gain
    - class_weight dict, list of dict or “balanced”, default=None
        - Weights associated with classes in the form {class_label: weight}. If None, all classes are supposed to have weight one. 
        For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
        - Note that for multioutput (including multilabel) weights should be defined for each class of every column in its own dict. 
        For example, for four-class multilabel classification weights should be [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] 
        instead of [{1:1}, {2:5}, {3:1}, {4:1}].
        - The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies 
        in the input data as n_samples / (n_classes * np.bincount(y))
        - For multi-output, the weights of each column of y will be multiplied.
        - Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
    
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
# Applying DecisionTreeClassifier Model
DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='entropy', 
                                                     splitter='best',
                                                     max_depth=3, 
                                                     class_weight= "balanced",
                                                     random_state=33) 
DecisionTreeClassifierModel.fit(X_train, y_train)
# ----------------------------------------------------
# Calculating Details
print('DecisionTreeClassifierModel Train Score is : ', DecisionTreeClassifierModel.score(X_train, y_train))
print('DecisionTreeClassifierModel Test Score is : ', DecisionTreeClassifierModel.score(X_test, y_test))
print("="*10)
# ---------------
print('DecisionTreeClassifierModel Classes are : ', DecisionTreeClassifierModel.classes_)  
print('DecisionTreeClassifierModel The number of classes are : ', DecisionTreeClassifierModel.n_classes_)  
print("The number of outputs when fit is performed : ",DecisionTreeClassifierModel.n_outputs_)
print()
print("The number of features when fit is performed : ",DecisionTreeClassifierModel.n_features_)
print("The inferred value of max_features : ",DecisionTreeClassifierModel.max_features_)
print("the feature importances : ",DecisionTreeClassifierModel.feature_importances_)
print()
print("="*10)
# ----------------------------------------------------
# Calculating Prediction
y_pred = DecisionTreeClassifierModel.predict(X_test)
y_pred_prob = DecisionTreeClassifierModel.predict_proba(X_test)
print('Prediction Probabilities Value for DecisionTreeClassifierModel is : \n', y_pred_prob[:5])
print('Pred Value for DecisionTreeClassifierModel is : ', y_pred[:5])
print('True Value for DecisionTreeClassifierModel is : ' , y_test[:5])
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
Z = DecisionTreeClassifierModel.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

plt.figure()
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1);
plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
plt.show(block=False) 

# plt.figure('Decision Tree')
# sklearn_tree.plot_tree(DecisionTreeClassifierModel)
# plt.show(block=False)


model = DecisionTreeClassifierModel
plt.figure("Feature importance")
plt.barh(range(model.n_features_), model.feature_importances_, align='center')
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.yticks(np.arange(model.n_features_), np.arange(model.n_features_)+1)
plt.ylim(-1, model.n_features_)
plt.show(block=False) 


plt.show(block=True) 