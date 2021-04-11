# Import Libraries
from sklearn.metrics import precision_score
# ----------------------------------------------------
'''
sklearn.metrics.precision_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')

'''
# Calculating Precision Score : (Specificity) #(TP / float(TP + FP))
# precision_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’,sample_weight=None)

y_true = ['a', 'b', 'b', 'a', 'b', 'a', 'a', 'b', 'a', 'b']
y_pred = ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'a', 'a', 'a']

# it can be : binary,macro,weighted,samples
PrecisionScore = precision_score(y_true, y_pred, average='micro')
print('Precision Score is : ', PrecisionScore)
