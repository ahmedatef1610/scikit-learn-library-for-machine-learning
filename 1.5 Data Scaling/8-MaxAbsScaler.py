from sklearn.preprocessing import MaxAbsScaler
X = [[1., 10., 2.],
     [2., 0., 0.],
     [5., 1., -1.]]
transformer = MaxAbsScaler().fit(X)
X = transformer.transform(X)
print(X)
# [[ 0.2  1.   1. ]
#  [ 0.4  0.   0. ]
#  [ 1.   0.1 -0.5]]