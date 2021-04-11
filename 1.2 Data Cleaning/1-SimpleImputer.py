# Import Libraries

from sklearn.impute import SimpleImputer
import numpy as np

# ----------------------------------------------------


# Cleaning data

'''
impute.SimpleImputer(missing_values=nan, strategy='mean', fill_value=None, verbose=0, copy=True, add_indicator=False)

strategy default='mean'
strategy => mean , median , most_frequent , constant
'''
X = [[1, 2, 0],
     [3, 0, 1],
     [5, 0, 0],
     [0, 4, 6],
     [5, 0, 0],
     [4, 5, 5]]

ImputedModule = SimpleImputer(missing_values=np.nan, strategy='mean')
ImputedX = ImputedModule.fit(X)
X = ImputedX.transform(X)


# X Data
print(X)

