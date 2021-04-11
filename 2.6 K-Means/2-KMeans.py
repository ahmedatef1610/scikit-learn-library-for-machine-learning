# Import Libraries
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
# creating data
X = np.random.rand(10000, 2)
y = np.random.rand(10000)
print(X.shape,y.shape)
print("="*25)
# ----------------------------------------------------
# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------
# Applying KMeans Model
KMeansModel = KMeans(n_clusters=5, init='k-means++', random_state=33, algorithm='auto') 
KMeansModel.fit(X_train)
# ----------------------------------------------------
# Calculating Details
print('KMeansModel Train Score is : ', KMeansModel.score(X_train))
print('KMeansModel Test Score is : ', KMeansModel.score(X_test))
print('KMeansModel centers are : ', KMeansModel.cluster_centers_)
print('KMeansModel No. of iteration is : ', KMeansModel.n_iter_)
print('KMeansModel intertia is : ', KMeansModel.inertia_)
print('KMeansModel labels are : ', KMeansModel.labels_)
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = KMeansModel.predict(X_test)
print('Pred Value for KMeansModel is : ', y_pred[:5])
print('True Value for KMeansModel is : ', y_test[:5])
# ----------------------------------------------------
plt.figure()
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1);
plt.show(block=True) 
