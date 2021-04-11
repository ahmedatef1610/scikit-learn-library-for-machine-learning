# Import Libraries
from sklearn.datasets import load_digits
#----------------------------------------------------

# load digits data

DigitsData = load_digits()

# X Data
X = DigitsData.data
print('X Data is ' , X[:10])
print('X shape is ' , X.shape)
print('X Features are ' , DigitsData.feature_names)

# y Data
y = DigitsData.target
print('y Data is ' , y[:10])
print('y shape is ' , y.shape)
print('y Columns are ' , DigitsData.target_names)




import matplotlib.pyplot as plt
plt.gray()

# for g in range(3):
#     print('Images of Number : ' , g)
#     plt.imshow(DigitsData.images[g])
#     print('------------------------------')
#     plt.show()

plt.imshow(X[16].reshape((8,8)))
print('y Data is ' , y[16])
plt.show()
