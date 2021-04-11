# Import Libraries
from sklearn.metrics import recall_score
# ----------------------------------------------------
'''
sklearn.metrics.recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')


'''
# Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))   1 / 1+2
# recall_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)

y_true = ['a', 'b', 'b', 'a', 'b', 'a', 'a', 'b', 'a', 'b']
y_pred = ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'a', 'a', 'a']

# it can be : binary,macro,weighted,samples
RecallScore = recall_score(y_true, y_pred, average='micro')
print()
print('Recall Score is : ', RecallScore)
