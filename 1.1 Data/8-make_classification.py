#Import Libraries
from sklearn.datasets import make_classification
#----------------------------------------------------

#load classification data

'''
X, y = make_classification(n_samples = 100, n_features = 20, n_informative = 2, n_redundant = 2,
                           n_repeated = 0, n_classes = 2, n_clusters_per_class = 2, weights = None,
                           flip_y = 0.01, class_sep = 1.0, hypercube = True, shift = 0.0,
                           Scale() = 1.0, shuffle = True, random_state = None)

- X[:, :n_informative + n_redundant + n_repeated]
- ValueError: Number of informative, redundant and repeated features must sum to less than the number of total features
- ValueError: n_classes(2) * n_clusters_per_class(5) must be smaller or equal 2**n_informative(2)=4
- len(weights) == n_classes - 1  Note that the actual class proportions will not exactly match weights when flip_y isn’t 0.
- flip_y > 0 might lead to less than n_classes in y in some cases.
'''

X, y = make_classification(n_samples = 100, n_features = 20, shuffle = True)

#X Data
print('X Data is \n' , X[:10])
print('X shape is ' , X.shape)

#y Data
print('y Data is \n' , y[:10])
print('y shape is ' , y.shape)