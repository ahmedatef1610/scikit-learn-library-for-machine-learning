# Import Libraries
from sklearn.metrics import accuracy_score
# ----------------------------------------------------
'''
sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)

normalizebool, default=True
If False, return the number of correctly classified samples. Otherwise, return the fraction of correctly classified samples.

'''
y_true = ['a', 'b', 'b', 'a', 'b', 'a', 'a', 'b', 'a', 'b']
y_pred = ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'a', 'a', 'a']

# Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))
AccScore = accuracy_score(y_true, y_pred)
print('Accuracy Score is : ', AccScore)
