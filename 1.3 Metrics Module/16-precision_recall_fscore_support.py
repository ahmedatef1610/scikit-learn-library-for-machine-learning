# Import Libraries
from sklearn.metrics import precision_recall_fscore_support
# ----------------------------------------------------
'''

sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=None, pos_label=1, average=None, warn_for='precision', 'recall', 'f-score', sample_weight=None, zero_division='warn')


'''
# Calculating Precision recall Score :
# metrics.precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=None, pos_label=1, average=
#                                        None, warn_for = ('precision’,’recall’, ’f-score’), sample_weight=None)

y_true = ['a', 'b', 'b', 'a', 'b', 'a', 'a', 'b', 'a', 'b']
y_pred = ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'a', 'a', 'a']

PrecisionRecallScore = precision_recall_fscore_support(y_true, y_pred, average='micro')  # it can be : binary,macro,weighted,samples
print('Precision Recall Score is : ', PrecisionRecallScore)
