# Import Libraries
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
# ----------------------------------------------------
'''
sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)

'''
# ----------------------------------------------------
# Splitting data

X , y = make_regression(n_samples=100, n_features=2, shuffle=True)
# showing data
# print('X \n', X[:5])
# print('y \n', y[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)

# Splitted Data
print('X_train shape is ', X_train.shape)
print('X_test shape is ', X_test.shape)
print('y_train shape is ', y_train.shape)
print('y_test shape is ', y_test.shape)
