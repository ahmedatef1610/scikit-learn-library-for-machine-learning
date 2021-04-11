# Import Libraries
from sklearn.metrics import classification_report
# ----------------------------------------------------
'''
sklearn.metrics.classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')


'''
# Calculating classification Report :
# classification_report(y_true, y_pred, labels=None, target_names=None,sample_weight=None, digits=2, output_dict=False)

y_true = ['a', 'b', 'b', 'a', 'b', 'a', 'a', 'b', 'a', 'b']
y_pred = ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'a', 'a', 'a']

ClassificationReport = classification_report(y_true, y_pred)

print(ClassificationReport)
