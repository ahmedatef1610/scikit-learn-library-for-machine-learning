import numpy as np
from sklearn.model_selection import ShuffleSplit
# ----------------------------------------------------
'''

class sklearn.model_selection.ShuffleSplit(n_splits=10, test_size=None, train_size=None, random_state=None)

'''
# ----------------------------------------------------
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
y = np.array([1, 2, 1, 2, 1, 2])

rs = ShuffleSplit(n_splits=5, test_size=.1, random_state=0)

print(rs.get_n_splits(X))
print(rs)

for train_index, test_index in rs.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

rs = ShuffleSplit(n_splits=5, train_size=0.5, test_size=.25,random_state=0)
for train_index, test_index in rs.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
