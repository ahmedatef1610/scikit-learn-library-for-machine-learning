#Import Libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
# ----------------------------------------------------
'''
sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, 
                                        metric='minkowski', metric_params=None, n_jobs=None, **kwargs)
===

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
# Applying KNeighborsClassifier Model 

KNNClassifierModel = KNeighborsClassifier(n_neighbors=5,
                                          weights='uniform',
                                          algorithm='auto',
                                          p=2,
                                          n_jobs=-1,
                                          )
KNNClassifierModel.fit(X_train, y_train)
# ----------------------------------------------------
#Calculating Details
print('KNNClassifierModel Train Score is : ' , KNNClassifierModel.score(X_train, y_train))
print('KNNClassifierModel Test Score is : ' , KNNClassifierModel.score(X_test, y_test))
print("="*10)
# ----------------------
print('The distance metric to use is : ' , KNNClassifierModel.effective_metric_)
print('Additional keyword arguments for the metric function is : ' , KNNClassifierModel.effective_metric_params_)
print('Number of samples in the fitted data is : ' , KNNClassifierModel.n_samples_fit_)
print('False when y’s shape is (n_samples, ) or (n_samples, 1) during fit otherwise True : ' , KNNClassifierModel.outputs_2d_)
print('Class labels known to the classifier is : ' , KNNClassifierModel.classes_)

print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = KNNClassifierModel.predict(X_test)
y_pred_prob = KNNClassifierModel.predict_proba(X_test)
print('Prediction Probabilities Value for KNNClassifierModel is : \n', y_pred_prob[:5])
print('Pred Value for KNNClassifierModel is : ', y_pred[:5])
print('True Value for KNNClassifierModel is : ' , y_test[:5])
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
Z = KNNClassifierModel.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

plt.figure()
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1);
plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
plt.show(block=False) 
# ------------------------
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 21 to good selection num neighbors
neighbors_settings = range(1, 40)
for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors,weights='uniform',algorithm = 'auto',p = 2,n_jobs=-1,)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
plt.figure('choose', figsize=(10,7))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.xticks(np.arange(0,40,1))
plt.legend()
plt.tight_layout()
plt.show(block=False)

# ------------------------


plt.show(block=True) 