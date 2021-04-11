# Import Libraries
from sklearn.metrics import r2_score
# ----------------------------------------------------
'''
sklearn.metrics.r2_score(y_true, y_pred, sample_weight=None, multioutput='uniform_average')

multioutput{‘raw_values’, ‘uniform_average’, ‘variance_weighted’}, array-like of shape (n_outputs,) 
or None, default=’uniform_average’

'''

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
r2 = r2_score(y_true, y_pred)
print(r2)  # 0.5

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
r2 = r2_score(y_true, y_pred)
print(r2)

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
r2 = r2_score(y_true, y_pred, multioutput='variance_weighted')
print(r2)

y_true = [1, 2, 3]
y_pred = [1, 2, 3]
r2 = r2_score(y_true, y_pred)
print(r2)

y_true = [1, 2, 3]
y_pred = [2, 2, 2]
r2 = r2_score(y_true, y_pred)
print(r2)

y_true = [1, 2, 3]
y_pred = [3, 2, 1]
r2 = r2_score(y_true, y_pred)
print(r2)
