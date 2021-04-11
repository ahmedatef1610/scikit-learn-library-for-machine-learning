# Import Libraries
from sklearn.datasets import make_regression
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# ----------------------------------------------------
'''

class sklearn.preprocessing.LabelEncoder

'''
# ----------------------------------------------------

# x = [1, 2, 5, 5, 5, 6]

# le = LabelEncoder()
# le.fit(x)

# print(le.classes_) # [1 2 5 6]

# print(le.transform(x)) # [0 1 2 2 2 3]

# print(le.inverse_transform([0, 1, 2, 2, 2, 3])) # [1, 2, 5, 5, 5, 6]

# ----------------------------------------------------
# x = ["paris", "paris", "tokyo", "amsterdam"]

# le = LabelEncoder()

# le.fit(x)

# print(le.classes_) # ['amsterdam' 'paris' 'tokyo']

# print(le.transform(x)) # [1 1 2 0]

# print(le.inverse_transform([2, 2, 1])) # ['tokyo' 'tokyo' 'paris']
# ----------------------------------------------------
# Importing the dataset
dataset = pd.read_csv('path/1.5 Data Scaling/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(y)
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
print(y)

print("="*25)

print(X)
label_encoder_X = LabelEncoder()
X[:,0] = label_encoder_X.fit_transform(X[:,0])
print(X)
# ----------------------------------------------------
