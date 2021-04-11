# Import Libraries
from sklearn.metrics import roc_auc_score
# ----------------------------------------------------
'''
sklearn.metrics.roc_auc_score(y_true, y_score, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)

'''
# Calculating ROC AUC Score:
# roc_auc_score(y_true, y_score, average=’macro’, sample_weight=None,max_fpr=None)

y_true = [1, 0, 1, 1, 1, 0, 0, 0]
y_pred = [1, 0, 0, 1, 1, 1, 0, 1]

# it can be : macro,weighted,samples
ROC_AUC_Score = roc_auc_score(y_true, y_pred, average='micro')
print('ROC_AUC_Score Score : ', ROC_AUC_Score)
