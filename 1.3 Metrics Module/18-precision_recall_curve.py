# Import Libraries
from sklearn.metrics import precision_recall_curve
# ----------------------------------------------------
'''

sklearn.metrics.precision_recall_curve(y_true, probas_pred, pos_label=None, sample_weight=None)

'''
# Calculating Precision recall Curve :
# precision_recall_curve(y_true, probas_pred, pos_label=None, sample_weight=None)

y_true = [0, 0, 1, 1]
y_pred = [0.1, 0.4, 0.35, 0.8]

PrecisionValue, RecallValue, ThresholdsValue = precision_recall_curve(y_true, y_pred)
print('Precision Value is : ', PrecisionValue)
print('Recall Value is : ', RecallValue)
print('Thresholds Value is : ', ThresholdsValue)
