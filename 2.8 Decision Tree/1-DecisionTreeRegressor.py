# Import Libraries

from sklearn.tree import DecisionTreeRegressor
import sklearn.tree as sklearn_tree

from sklearn.datasets import make_blobs, make_classification, make_regression, load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
'''
# A decision tree regressor

class sklearn.tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                                            min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
                                            min_impurity_decrease=0.0, min_impurity_split=None, ccp_alpha=0.0)


===
    - criterion{“mse”, “friedman_mse”, “mae”, “poisson”}, default=”mse”
        The function to measure the quality of a split. Supported criteria are “mse” for the mean squared error, 
        which is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of each terminal node, 
        “friedman_mse”, which uses mean squared error with Friedman’s improvement score for potential splits, 
        “mae” for the mean absolute error, which minimizes the L1 loss using the median of each terminal node, 
        and “poisson” which uses reduction in Poisson deviance to find splits.
    - splitter {“best”, “random”}, default=”best”
        The strategy used to choose the split at each node. 
        Supported strategies are “best” to choose the best split and “random” to choose the best random split.
    -max_depth int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or 
        until all leaves contain less than min_samples_split samples.
    - min_samples_split int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider min_samples_split as the minimum number.
        - If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) 
            are the minimum number of samples for each split.
    - min_samples_leaf int or float, default=1
        The minimum number of samples required to be at a leaf node. 
        A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples 
        in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
        - If int, then consider min_samples_leaf as the minimum number.
        - If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
    - min_weight_fraction_leaf float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. 
        Samples have equal weight when sample_weight is not provided.
    - max_features int, float or {“auto”, “sqrt”, “log2”}, default=None
        The number of features to consider when looking for the best split:
        If int, then consider max_features features at each split.
        If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
        If “auto”, then max_features=n_features.
        If “sqrt”, then max_features=sqrt(n_features).
        If “log2”, then max_features=log2(n_features).
        If None, then max_features=n_features.
        Note: the search for a split does not stop until at least one valid partition of the node samples is found, 
        even if it requires to effectively inspect more than max_features features.
    - max_leaf_nodes int, default=None
        Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. 
        If None then unlimited number of leaf nodes.
    - min_impurity_decrease float, default=0.0
        A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
        The weighted impurity decrease equation is the following:
        N_t / N * (impurity - N_t_R / N_t * right_impurity
                            - N_t_L / N_t * left_impurity)
        where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, 
            and N_t_R is the number of samples in the right child.
        N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.
    - min_impurity_split float, default=0
        Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.
    ccp_alpha non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. 
        The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed. 
        See Minimal Cost-Complexity Pruning for details.
===

'''
# ----------------------------------------------------
X, y = make_regression(n_samples=10000, n_features=1, shuffle=True, noise=25, random_state=16)
print(X.shape, y.shape)
print("="*10)
# ---------------
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
# ---------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=44, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------
# Applying DecisionTreeRegressor Model
DecisionTreeRegressorModel = DecisionTreeRegressor(criterion='mae',
                                                   splitter='best',
                                                   max_depth=5,
                                                   max_features=None, 
                                                   min_samples_split=2,
                                                   min_samples_leaf=1,
                                                   max_leaf_nodes=None,
                                                   min_impurity_decrease=0.0,
                                                   ccp_alpha=0.0,
                                                   random_state=33,
                                                   )
DecisionTreeRegressorModel.fit(X_train, y_train)
# ----------------------------------------------------
# Calculating Details
print('DecisionTreeRegressorModel Train Score is : ', DecisionTreeRegressorModel.score(X_train, y_train))
print('DecisionTreeRegressorModel Test Score is : ', DecisionTreeRegressorModel.score(X_test, y_test))
print("="*10)
# ------------------
print("The number of outputs when fit is performed : ",DecisionTreeRegressorModel.n_outputs_)
print()
print("The number of features when fit is performed : ",DecisionTreeRegressorModel.n_features_)
print("The inferred value of max_features : ",DecisionTreeRegressorModel.max_features_)
print("the feature importances : ",DecisionTreeRegressorModel.feature_importances_)
print()
# print("The underlying Tree object : ",DecisionTreeRegressorModel.tree_)
# print(DecisionTreeRegressorModel.tree_.node_count)
# print(DecisionTreeRegressorModel.tree_.children_left)
# print(DecisionTreeRegressorModel.tree_.children_right)
# print(DecisionTreeRegressorModel.tree_.feature)
# print(DecisionTreeRegressorModel.tree_.threshold)
print()
print("="*10)
# ------------------
# Calculating Prediction
y_pred = DecisionTreeRegressorModel.predict(X_test)
print('Pred Value for DecisionTreeRegressorModel is : ', y_pred[:5])
print('True Value for DecisionTreeRegressorModel is : ', y_test[:5])
print("="*25)
# ----------------------------------------------------
x_axis = np.arange(0,1,0.001)
x_axis = x_axis.reshape(-1,1)
print(x_axis.shape)

# plt.figure('Decision Tree')
# sklearn_tree.plot_tree(DecisionTreeRegressorModel)
# plt.show(block=False)
# plt.imsave("path/2.8 Decision Tree/Decision_Tree.png", sklearn_tree.plot_tree(DecisionTreeRegressorModel))
# print(sklearn_tree.plot_tree(DecisionTreeRegressorModel))

plt.figure('Decision Tree Regressor')
sns.scatterplot(x=X[:,0], y=y, alpha=0.5)
sns.lineplot(x=x_axis[:,0], y=DecisionTreeRegressorModel.predict(x_axis), color='k')
plt.show(block=False) 

model = DecisionTreeRegressorModel
plt.figure("Feature importance")
plt.barh(range(model.n_features_), model.feature_importances_, align='center')
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.yticks(np.arange(model.n_features_), np.arange(model.n_features_)+1)
plt.ylim(-1, model.n_features_)
plt.show(block=True) 


plt.show(block=True) 
