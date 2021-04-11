# Import Libraries
from sklearn.datasets import make_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import numpy as np
import pandas as pd

# ----------------------------------------------------
'''
class sklearn.preprocessing.OneHotEncoder(categories='auto', drop=None, sparse=True, 
                                            dtype=<class 'numpy.float64'>, handle_unknown='error')
--------
categories ‘auto’ or a list of array-like, default=’auto’
--
drop {‘first’, ‘if_binary’} or a array-like of shape (n_features,), default=None
--
sparse bool, default=True
Will return sparse matrix if set True else will return an array.
--
dtype number type, default=float
Desired dtype of output.
--
handle_unknown {‘error’, ‘ignore’}, default=’error’
--
'''
# ----------------------------------------------------
'''
class sklearn.compose.ColumnTransformer(transformers, remainder='drop', sparse_threshold=0.3, n_jobs=None, 
transformer_weights=None, verbose=False)
---
transformers list of tuples
    - name str
    - transformer {‘drop’, ‘passthrough’} or estimator
    - columns str, array-like of str, int, array-like of int, array-like of bool, slice or callable
---
remainder {‘drop’, ‘passthrough’} or estimator, default=’drop’
---
sparse_threshold float, default=0.3
---
n_jobs int, default=None
---
transformer_weights dict, default=None
---
verbose bool, default=False

---

'''
# ----------------------------------------------------
# X = [['Male', 1], 
#      ['Female', 3], 
#      ['Female', 2]]


# # enc = OneHotEncoder(handle_unknown='ignore')
# enc = OneHotEncoder(handle_unknown='ignore')

# enc.fit(X)

# print(enc.categories_)
# print("="*25)

# print(enc.transform([['Female', 1], ['Male', 4]]).toarray())
# print("="*25)

# print(enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]]))
# print("="*25)

# print(enc.get_feature_names())
# print("="*25)

# print(enc.get_feature_names(['gender', 'group']))
# print("="*25)
# ----------------------------------------------------
# X = [['Male', 1], 
#      ['Female', 3], 
#      ['Female', 2]]

# drop_enc = OneHotEncoder(drop='first').fit(list(X))
# print(drop_enc.categories_)

# print(drop_enc.transform([['Female', 1], ['Male', 2]]).toarray())
# ----------------------------------------------------
# Importing the dataset
# dataset = pd.read_csv('path/1.5 Data Scaling/Data.csv')
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, -1].values

# print(X)
# print("="*25)

# label_encoder_X = LabelEncoder()
# X[:,0] = label_encoder_X.fit_transform(X[:,0])
# print(X)
# print("="*25)


# ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
# X = ct.fit_transform(X)
# print(X)
# print("="*25)

# one = pd.get_dummies(dataset['Country'])
# print(one)
# print(one.values)
# ----------------------------------------------------
# Importing the dataset
dataset = pd.read_csv('path/1.5 Data Scaling/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)
print("="*25)

enc = OneHotEncoder(handle_unknown='ignore')
x_country = enc.fit_transform(X[:,0:1]).toarray()

print('categories_ =>', enc.categories_)
print('get_feature_names =>', enc.get_feature_names(['Country']))
print("="*25)

print(X)
X = np.delete(X, [0], axis=1)
print(X)
print(x_country)
print(np.concatenate((x_country,X),axis=1))
print("="*25)
# -----------------------
# df = pd.concat([df, pd.DataFrame(stuff_cols, columns=enc.get_feature_names())], axis=1)
# print(df)
