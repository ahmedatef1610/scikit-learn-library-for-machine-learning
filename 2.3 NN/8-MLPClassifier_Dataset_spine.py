import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,confusion_matrix ,classification_report
# ----------------------------------------------------
df = pd.read_csv('path/2.3 NN/Dataset_spine.csv')
df = df.drop(['Unnamed: 13'], axis=1)
# print(df.head())
# print("="*10)
# print(df.info())
# print("="*10)
# print(df.describe())
# print("="*10)

# df = df.drop(['Col7','Col8','Col9','Col10','Col11','Col12'], axis=1)
# print(df.head())

X = df.drop(['Class_att'], axis=1)
y = df['Class_att']

print("="*25)
# ----------------------------------------------------
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
# -----------
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.25,random_state=27)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------
# Applying MLPClassifier Model
MLPClassifierModel = MLPClassifier( activation='tanh',
                                    solver='adam',
                                    learning_rate='adaptive',
                                    learning_rate_init=0.001,
                                    hidden_layer_sizes=(100,3), 
                                    max_iter=1000,
                                    batch_size = 5,
                                    random_state=33)

MLPClassifierModel.fit(X_train, y_train)
# 0.8205128205128205
# 0.8333333333333334
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
print("="*10)
plt.figure()
sns.lineplot(data=MLPClassifierModel.loss_curve_)
plt.show(block=False)
# ---------------------
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = MLPClassifierModel.predict(X_test)
y_pred_prob = MLPClassifierModel.predict_proba(X_test)
print('Prediction Probabilities Value for MLPClassifierModel is : ', y_pred_prob[:5])
print('Pred Value for MLPClassifierModel is : ', y_pred[:5])
print('True Value for MLPClassifierModel is : ' , y_test[:5].values)
print("="*10)
# ---------------------
from sklearn.metrics import confusion_matrix , classification_report

CM = confusion_matrix(y_test, y_pred)
ClassificationReport = classification_report(y_test, y_pred)
print(ClassificationReport)
print("="*10)
print(accuracy_score(y_test, y_pred))
print("="*10)

print(CM)
plt.figure()
sns.heatmap(CM, center = True, annot=True, fmt="d")
plt.show(block=True)

print("="*25)
# ----------------------------------------------------