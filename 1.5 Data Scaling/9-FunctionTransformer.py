# Import Libraries
from sklearn.datasets import make_regression
from sklearn.preprocessing import FunctionTransformer
# ----------------------------------------------------
'''

class sklearn.preprocessing.FunctionTransformer(func=None, inverse_func=None, validate=False, accept_sparse=False, 
check_inverse=True, kw_args=None, inv_kw_args=None)


'''
# ----------------------------------------------------
# Function Transforming Data

X ,y = make_regression(n_samples=500, n_features=3,shuffle=True)
# showing data
print('X \n', X[:5])
print('y \n', y[:5])

scaler = FunctionTransformer(func=lambda x: x**2, validate=True)  # or func = function1
X = scaler.fit_transform(X)

# showing data
print('X \n', X[:5])
print('y \n', y[:5])
