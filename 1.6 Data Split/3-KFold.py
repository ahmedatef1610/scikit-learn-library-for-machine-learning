# Import Libraries
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
# ----------------------------------------------------
'''
class sklearn.model_selection.KFold(n_splits=5, shuffle=False, random_state=None)

'''
# ----------------------------------------------------
# KFold Splitting data

X , y = make_regression(n_samples=100, n_features=2, shuffle=True)
# showing data
# print('X \n', X[:5])
# print('y \n', y[:5])

kf = KFold(n_splits=2, random_state=44, shuffle=True)

# KFold Data
for train_index, test_index in kf.split(X):
    print('Train Data is : \n', train_index)
    print('Test Data is  : \n', test_index)
    print('-------------------------------')
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('X_train Shape is  ', X_train.shape)
    print('X_test Shape is  ', X_test.shape)
    print('y_train Shape is  ', y_train.shape)
    print('y_test Shape is  ', y_test.shape)
    print('========================================')
