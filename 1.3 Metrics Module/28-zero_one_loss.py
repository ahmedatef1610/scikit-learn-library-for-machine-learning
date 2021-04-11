# Import Libraries
from sklearn.metrics import zero_one_loss
# ----------------------------------------------------
'''
sklearn.metrics.zero_one_loss(y_true, y_pred, normalize=True, sample_weight=None)

'''
# Calculating Zero One Loss:
# zero_one_loss(y_true, y_pred, normalize = True, sample_weight = None)

y_true = [1, 0, 1, 1, 1, 0, 0, 0]
y_pred = [1, 0, 0, 1, 1, 1, 0, 0]

ZeroOneLossValue = zero_one_loss(y_true, y_pred, normalize=False)
print('Zero One Loss Value : ', ZeroOneLossValue )
ZeroOneLossValue = zero_one_loss(y_true, y_pred, normalize=True)
print('Zero One Loss Value : ', ZeroOneLossValue )
