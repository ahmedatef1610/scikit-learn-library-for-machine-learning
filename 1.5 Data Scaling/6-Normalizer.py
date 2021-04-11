from sklearn.preprocessing import Normalizer

X = [[4, 1, 2, 2], 
     [1, 3, 9, 3], 
     [5, 7, 5, 1]]


#transformer = Normalizer(norm='l1' )

#transformer = Normalizer(norm='l2' )

transformer = Normalizer(norm='max')

transformer.fit(X)
X = transformer.transform(X)
print(X)
# [[1.         0.25       0.5        0.5       ]
#  [0.11111111 0.33333333 1.         0.33333333]
#  [0.71428571 1.         0.71428571 0.14285714]]