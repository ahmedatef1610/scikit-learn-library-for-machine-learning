# Import Libraries
from sklearn.metrics import f1_score
# ----------------------------------------------------
'''
sklearn.metrics.f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')



'''

# Calculating F1 Score  : 2 * (precision * recall) / (precision + recall)
# f1_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)
# it can be : binary,micro,macro,weighted,samples

y_true = ['a', 'b', 'b', 'a', 'b', 'a', 'a', 'b', 'a', 'b']
y_pred = ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'a', 'a', 'a']

F1Score = f1_score(y_true, y_pred, average='micro')
print('F1 Score is : ', F1Score)
