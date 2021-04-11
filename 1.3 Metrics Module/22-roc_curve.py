# Import Libraries
from sklearn.metrics import roc_curve
# ----------------------------------------------------

'''
sklearn.metrics.roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)

pos_label int or str, default=None
The label of the positive class. When pos_label=None, if y_true is in {-1, 1} or {0, 1},
pos_label is set to 1, otherwise an error will be raised.

'''

# Calculating Receiver Operating Characteristic :
# roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)

y_true = [1,0,1,1,1,0,0,0]
y_pred = [1,1,0,1,1,1,0,1]

fprValue, tprValue, thresholdsValue = roc_curve(y_true, y_pred)
print('fpr Value  : ', fprValue)
print('tpr Value  : ', tprValue)
print('thresholds Value  : ', thresholdsValue)
