#Import Libraries
from sklearn.datasets import make_regression
from sklearn.preprocessing import Normalizer
#----------------------------------------------------
'''
work on rows

class sklearn.preprocessing.Normalizer(norm='l2', copy=True)

norm {‘l1’, ‘l2’, ‘max’}, default=’l2’
The norm to use to normalize each non zero sample. 
If norm=’max’ is used, values will be rescaled by the maximum of the absolute values.

تستخدم l1  لجعل مجموع كل صف هو القيمة العظمي 
تستخدم l2 لجعل جذر مجموع مربعات كل صف هو القيمة العظمي 
تستخدم max   لجعل القيمة العظمي في كل صف هي القيمة العظمي


'''
# ----------------------------------------------------
#Normalizing Data

X ,y = make_regression(n_samples=500, n_features=3,shuffle=True)
# showing data
print('X \n', X[:5])
print('y \n', y[:5])

scaler = Normalizer(copy=True, norm='l2') # you can change the norm to 'l1' or 'max' 
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:5])
print('y \n' , y[:5])