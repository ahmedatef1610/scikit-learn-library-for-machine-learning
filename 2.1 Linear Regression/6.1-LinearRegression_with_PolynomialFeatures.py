import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('path/2.1 Linear Regression/satf.csv')
print(dataset.head(10))
print("="*25)
X = dataset.iloc[:, :1]
y = dataset.iloc[:, -1]
print(X)
print("="*25)
print(y)
print("="*25)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print("="*25)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)

X_train2 = poly_reg.fit_transform(X_train)
X_test2 = poly_reg.fit_transform(X_test)

# No Polynomial for y
print(X_train2.shape)
print(X_test2.shape)

from sklearn.linear_model import LinearRegression
lin_reg_2 = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=-1)
lin_reg_2.fit(X_train2, y_train )


print(lin_reg_2.score(X_train2, y_train))
print(lin_reg_2.score(X_test2, y_test))
print("="*25)

# Predicting the Test set results
y_pred2 = lin_reg_2.predict(X_test2)
print("Pred ",y_pred2[:5])
print("True ",y_test[:5].values)
print("="*25)

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred2))

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred2))

from sklearn.metrics import median_absolute_error
print(median_absolute_error(y_test, y_pred2))
print("="*25)



 
# Visualising the Training set results
#plt.scatter(X_train, y_train, color = 'red')
# plt.scatter(X_test, y_test, color = 'green')
plt.scatter(X, y, color = 'green')
plt.plot(X_train, lin_reg_2.predict(poly_reg.fit_transform(X_train)), color = 'blue')
# plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('SAT degrees')
plt.xlabel('high_GPA')
plt.ylabel('univ_GPA')
plt.show()

 
 
