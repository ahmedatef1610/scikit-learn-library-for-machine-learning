# Import Libraries
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
# ----------------------------------------------------
'''
sklearn.metrics.roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
sklearn.metrics.auc(x, y)
'''
# Calculating Area Under the Curve :

y_true = [1, 0, 1, 1, 1, 0, 0, 0]
y_pred = [1, 0, 1, 1, 1, 1, 0, 1]

fpr, tpr, thresholds = roc_curve(y_true, y_pred)
auc = auc(fpr, tpr)
print('AUC Value  : ', auc)
