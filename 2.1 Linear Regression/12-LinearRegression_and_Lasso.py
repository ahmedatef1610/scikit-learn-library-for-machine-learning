from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
X = cancer.data
Y = cancer.target
# print(cancer.keys()) # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
# print(cancer.target_names) # ['malignant' 'benign']
# print(cancer_df.head())
# print(X[:,:3])
# print(Y[:3])
print("="*50)

##################################################################
# splitting
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=31)
##################################################################
# apply linear regression

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_train_score = lr.score(X_train, y_train)
lr_test_score = lr.score(X_test, y_test)
print("LR training score:", lr_train_score)
print("LR test score: ", lr_test_score)
print("="*50)

##################################################################
# apply lasso

lasso = Lasso()
lasso.fit(X_train, y_train)
train_score = lasso.score(X_train, y_train)
test_score = lasso.score(X_test, y_test)

coeff_used = np.sum(lasso.coef_ != 0)
coeff_used = np.count_nonzero(lasso.coef_ != 0)
coeff_used = len(lasso.coef_[lasso.coef_!=0])

print("training score:", train_score)
print("test score: ", test_score)
print("number of features : ", len(lasso.coef_))
print("number of features used: ", coeff_used)
print("="*50)

lasso001 = Lasso(alpha=0.01, max_iter=10e5)
lasso001.fit(X_train, y_train)
train_score001 = lasso001.score(X_train, y_train)
test_score001 = lasso001.score(X_test, y_test)
coeff_used001 = len(lasso001.coef_[lasso001.coef_!=0])
print("training score for alpha = 0.01:", train_score001)
print("test score for alpha = 0.01: ", test_score001)
print("number of features used: for alpha = 0.01:", coeff_used001)
print("="*50)

lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
lasso00001.fit(X_train, y_train)
train_score00001 = lasso00001.score(X_train, y_train)
test_score00001 = lasso00001.score(X_test, y_test)
coeff_used00001 = len(lasso00001.coef_[lasso00001.coef_!=0])

print("training score for alpha = 0.0001:", train_score00001)
print("test score for alpha = 0.0001: ", test_score00001)
print("number of features used: for alpha = 0.0001:", coeff_used00001)
print("="*50)

# ##################################################################
# graph

plt.subplot(1,2,1)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7)
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$')

plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)



plt.subplot(1,2,2)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7)
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$')
plt.plot(lasso00001.coef_,alpha=0.8,linestyle='none',marker='v',markersize=6,color='black',label=r'Lasso; $\alpha = 0.00001$')
plt.plot(lr.coef_,alpha=0.7,linestyle='none',marker='o',markersize=5,color='green',label='Linear Regression',zorder=2)

plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)

plt.tight_layout()
plt.show()
