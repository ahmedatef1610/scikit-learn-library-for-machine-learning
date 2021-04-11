import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_boston

# load data
boston = load_boston()
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
# add another column that contains the house prices which in scikit learn datasets are considered as target
boston_df['Price'] = boston.target
print(boston_df.info())
print("="*10)
print(boston_df.head())
newX = boston_df.drop('Price', axis=1)
newY = boston_df['Price']
# print(newX[0:3])  # check
# print(type(newY)) # pandas core frame
print("="*50)
##################################################################
# split
X_train, X_test, y_train, y_test = train_test_split(newX, newY, test_size=0.3, random_state=3)
print(X_train.shape," - ",X_test.shape)
print(y_train.shape," - ",y_test.shape)
print("="*50)
##################################################################
# apply linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
# apply ridge
rr = Ridge(alpha=0.01)
rr.fit(X_train, y_train)
# apply ridge 100
rr100 = Ridge(alpha=100)  # comparison with alpha value
rr100.fit(X_train, y_train)
##################################################################
train_score = lr.score(X_train, y_train)
test_score = lr.score(X_test, y_test)
Ridge_train_score = rr.score(X_train, y_train)
Ridge_test_score = rr.score(X_test, y_test)
Ridge_train_score100 = rr100.score(X_train, y_train)
Ridge_test_score100 = rr100.score(X_test, y_test)

print("linear regression train score:", train_score)
print("linear regression test score:", test_score)
print("ridge regression train score low alpha:", Ridge_train_score)
print("ridge regression test score low alpha:", Ridge_test_score)
print("ridge regression train score high alpha:", Ridge_train_score100)
print("ridge regression test score high alpha:", Ridge_test_score100)
##################################################################
# graph
plt.plot(rr.coef_, alpha=0.7, linestyle='none', marker='*', markersize=5, color='red', label=r'Ridge; $\alpha = 0.01$', zorder=7)  
plt.plot(rr100.coef_, alpha=0.5, linestyle='none', marker='d', markersize=6, color='blue', label=r'Ridge; $\alpha = 100$')
plt.plot(lr.coef_, alpha=0.4, linestyle='none', marker='o', markersize=7, color='green', label='Linear Regression')

plt.xlabel('Coefficient Index', fontsize=16)
plt.ylabel('Coefficient Magnitude', fontsize=16)
plt.legend(fontsize=13, loc=4)
plt.show()
