from sklearn.metrics import mean_absolute_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]


MAE = mean_absolute_error(y_true, y_pred)
print(MAE)  # 0.5

y_true = [[0.5, 1],
          [-1, 1],
          [7, -6]]
y_pred = [[0, 2],
          [-1, 2],
          [8, -5]]


MAE = mean_absolute_error(y_true, y_pred)
print(MAE)  # 0.75
MAE = mean_absolute_error(y_true, y_pred, multioutput='uniform_average')
print(MAE)  # 0.75

MAE = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
print(MAE)  # array([0.5, 1. ])
