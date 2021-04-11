#Import Libraries
from sklearn.metrics import median_absolute_error
#----------------------------------------------------
'''
sklearn.metrics.median_absolute_error(y_true, y_pred, multioutput='uniform_average', sample_weight=None)

multioutput{‘raw_values’, ‘uniform_average’} or array-like of shape (n_outputs,), default=’uniform_average’

'''
# y_pred = []
# y_test= []
# #Calculating Median Squared Error
# MdSEValue = median_absolute_error(y_test, y_pred)
# #print('Median Squared Error Value is : ', MdSEValue )

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
MdSE = median_absolute_error(y_true, y_pred)
print(MdSE) # 0.5
