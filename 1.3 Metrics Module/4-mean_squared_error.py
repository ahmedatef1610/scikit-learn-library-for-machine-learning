from sklearn.metrics import mean_squared_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

MSE = mean_squared_error(y_true, y_pred)
print(MSE)

y_true = [[0.5, 1],
          [-1, 1],
          [7, -6]]
y_pred = [[0, 2],
          [-1, 2],
          [8, -5]]


MSE = mean_squared_error(y_true, y_pred)
print(MSE)

MSE = mean_squared_error(y_true, y_pred, multioutput='uniform_average')
print(MSE)

MSE = mean_squared_error(y_true, y_pred, multioutput='raw_values')
print(MSE)
