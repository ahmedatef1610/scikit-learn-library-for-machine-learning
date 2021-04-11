# Import Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import zero_one_loss
from sklearn.metrics import roc_auc_score
# ----------------------------------------------------
# load breast cancer data
BreastData = load_breast_cancer()
# X Data
X = BreastData.data
# y Data
y = BreastData.target
# ----------------------------------------------------
# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
# ----------------------------------------------------
# Applying SGDClassifier Model
SGDClassifierModel = SGDClassifier(penalty='l2', loss='squared_loss', learning_rate='optimal', random_state=33)
SGDClassifierModel.fit(X_train, y_train)
# Calculating Details
print('SGDClassifierModel Train Score is : ', SGDClassifierModel.score(X_train, y_train))
print('SGDClassifierModel Test Score is : ', SGDClassifierModel.score(X_test, y_test))
print('SGDClassifierModel loss function is : ', SGDClassifierModel.loss_function_)
print('SGDClassifierModel No. of iterations is : ', SGDClassifierModel.n_iter_)
print('-'*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = SGDClassifierModel.predict(X_test)
print('Predicted Value for SGDClassifierModel is : ', y_pred[:5])
print('True Value for SGDClassifierModel is : ', y_test[:5])
# ----------------------------------------------------
# Calculating Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)
# drawing confusion matrix
sns.heatmap(CM, center=True)
plt.show()
# ----------------------------------------------------
# Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))
AccScore = accuracy_score(y_test, y_pred, normalize=False)
print('Accuracy Score is : ', AccScore)
# ----------------------------------------------------
# Calculating F1 Score  : 2 * (precision * recall) / (precision + recall)
# f1_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)
# it can be : binary,macro,weighted,samples
F1Score = f1_score(y_test, y_pred, average='micro')
print('F1 Score is : ', F1Score)
# ----------------------------------------------------
# Calculating Precision recall Score :
# metrics.precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=None, pos_label=1, average=
#                                        None, warn_for = ('precision’,’recall’, ’f-score’), sample_weight=None)
PrecisionRecallScore = precision_recall_fscore_support(y_test, y_pred, average='micro')  # it can be : binary,macro,weighted,samples
print('Precision Recall Score is : ', PrecisionRecallScore)
# ----------------------------------------------------
# Calculating Precision Score : (Specificity) #(TP / float(TP + FP))
# precision_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’,sample_weight=None)
# it can be : binary,macro,weighted,samples
PrecisionScore = precision_score(y_test, y_pred, average='micro')
print('Precision Score is : ', PrecisionScore)
# ----------------------------------------------------
# Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))   1 / 1+2
# recall_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)
# it can be : binary,macro,weighted,samples
RecallScore = recall_score(y_test, y_pred, average='micro')
print('Recall Score is : ', RecallScore)
# ----------------------------------------------------
# Calculating Precision recall Curve :
# precision_recall_curve(y_true, probas_pred, pos_label=None, sample_weight=None)
PrecisionValue, RecallValue, ThresholdsValue = precision_recall_curve(y_test, y_pred)
#print('Precision Value is : ', PrecisionValue)
#print('Recall Value is : ', RecallValue)
print('Thresholds Value is : ', ThresholdsValue)
# ----------------------------------------------------
# Calculating classification Report :
#classification_report(y_true, y_pred, labels=None, target_names=None,sample_weight=None, digits=2, output_dict=False)
ClassificationReport = classification_report(y_test, y_pred)
print('Classification Report is : ', ClassificationReport)
# ----------------------------------------------------
# Calculating Area Under the Curve :
fprValue2, tprValue2, thresholdsValue2 = roc_curve(y_test, y_pred)
AUCValue = auc(fprValue2, tprValue2)
print('AUC Value  : ', AUCValue)
# ----------------------------------------------------
# Calculating Zero One Loss:
#zero_one_loss(y_true, y_pred, normalize = True, sample_weight = None)
ZeroOneLossValue = zero_one_loss(y_test, y_pred, normalize=False)
print('Zero One Loss Value : ', ZeroOneLossValue)
# ----------------------------------------------------
# Calculating ROC AUC Score:
# roc_auc_score(y_true, y_score, average=’macro’, sample_weight=None,max_fpr=None)
# it can be : macro,weighted,samples
ROCAUCScore = roc_auc_score(y_test, y_pred, average='micro')
print('ROCAUC Score : ', ROCAUCScore)
# ----------------------------------------------------
# Calculating Receiver Operating Characteristic :
#roc_curve(y_true, y_score, pos_label=None, sample_weight=None,drop_intermediate=True)
fprValue, tprValue, thresholdsValue = roc_curve(y_test, y_pred)
print('fpr Value  : ', fprValue)
print('tpr Value  : ', tprValue)
print('thresholds Value  : ', thresholdsValue)
# ----------------------------------------------------
