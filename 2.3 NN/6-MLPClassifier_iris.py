# Import Libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
# load iris data
IrisData = load_iris()
# X Data
X = IrisData.data
# y Data
y = IrisData.target
print(IrisData.feature_names)
print(IrisData.target_names)
print("="*25)
# ----------------------------------------------------
# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------
# Applying MLPClassifier Model
MLPClassifierModel = MLPClassifier(activation='logistic',
                                   solver='adam',
                                   learning_rate='adaptive',
                                   learning_rate_init=0.001,
                                   hidden_layer_sizes=(100, 3),
                                   max_iter=1000,
                                   batch_size=5,
                                   random_state=33)
MLPClassifierModel.fit(X_train, y_train)
# 0.9666666666666667 => logistic Regression
# ----------------------------------------------------
# Calculating Details
print('MLPClassifierModel Train Score is : ', MLPClassifierModel.score(X_train, y_train))
print('MLPClassifierModel Test Score is : ',  MLPClassifierModel.score(X_test, y_test))
print("="*10)
# ---------------------
print("Class labels for each output : ", MLPClassifierModel.classes_)
print("Number of outputs : ", MLPClassifierModel.n_outputs_)
print('MLPClassifierModel last activation is : ' , MLPClassifierModel.out_activation_)
print('MLPClassifierModel No. of layers is : ' , MLPClassifierModel.n_layers_)
print('MLPClassifierModel No. of iterations is : ' , MLPClassifierModel.n_iter_)
print("The number of training samples seen by the solver during fitting : ", MLPClassifierModel.t_)
print("="*10)
# ---------------------
print('MLPClassifierModel loss is : ' , MLPClassifierModel.loss_)
print("MLPClassifierModel best loss is : ", MLPClassifierModel.best_loss_) # early_stopping = False (must be)
print("MLPClassifierModel loss curve is : ", MLPClassifierModel.loss_curve_[-5:]) 
print("MLPClassifierModel loss curve length is : ", len(MLPClassifierModel.loss_curve_)) 

plt.figure()
sns.lineplot(data=MLPClassifierModel.loss_curve_)
plt.show(block=False)
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = MLPClassifierModel.predict(X_test)
y_pred_prob = MLPClassifierModel.predict_proba(X_test)
print('Prediction Probabilities Value for MLPClassifierModel is : ', y_pred_prob[:5])
print('Predicted Value for MLPClassifierModel is : ', y_pred[:5])
print('True Value for MLPClassifierModel is : ' , y_test[:5])
print("="*25)
# ----------------------------------------------------
# Calculating Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)
# drawing confusion matrix
plt.figure()
sns.heatmap(CM, center=True, annot=True, fmt="d")
plt.show(block=True)
