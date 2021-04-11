# Import Libraries
from sklearn.metrics import mean_absolute_error 
#----------------------------------------------------
'''
sklearn.metrics.mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')

multioutput => {‘raw_values’, ‘uniform_average’} or array-like of shape (n_outputs,)

‘raw_values’ => 
‘uniform_average’ => each col
'''
y_pred = []
y_test= []

#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
#print('Mean Absolute Error Value is : ', MAEValue)