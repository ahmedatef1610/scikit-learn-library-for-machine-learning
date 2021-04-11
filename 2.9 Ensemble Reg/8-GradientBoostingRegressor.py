from sklearn.metrics import mean_squared_error
from sklearn.datasets import  make_friedman1
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
x , y = make_friedman1(n_samples = 1000, n_features=10, noise=1)
print(x.shape)
print(y.shape)
print("="*25)

# ----------------------------------------------------
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.3 )
# ----------------------------------------------------
model = GradientBoostingRegressor(n_estimators = 100 , learning_rate = 0.1 , max_depth = 3)
model.fit(x_train , y_train)
# ----------------------------------------------------
print(model.score(x_test , y_test))
print("="*25)
# ----------------------------------------------------
print(model.predict(x_test)[:5])
print("="*25)
print(mean_squared_error(y_test , model.predict(x_test)))
# ----------------------------------------------------
# x_axis = np.arange(0,1,0.001)
# x_axis = x_axis.reshape(-1,1)
# print(x_axis.shape)

# plt.figure()
# sns.scatterplot(x=x[:,0], y=y, alpha=0.5)
# # sns.lineplot(x=x_axis[:,0], y=model.predict(x_axis), color='k')
# plt.show(block=False) 

