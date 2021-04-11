#Import Libraries
from sklearn.metrics import mean_squared_error 
#----------------------------------------------------
'''
sklearn.metrics.mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average', squared=True)

squared bool, default=True
If True returns MSE value, if False returns RMSE value.

multioutput{‘raw_values’, ‘uniform_average’} or array-like of shape (n_outputs,), default=’uniform_average’

'''
y_pred = []
y_test= []
#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
#print('Mean Squared Error Value is : ', MSEValue)