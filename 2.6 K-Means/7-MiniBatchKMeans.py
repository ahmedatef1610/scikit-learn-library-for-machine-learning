# Import Libraries
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
# creating data
X = np.random.rand(10000, 2)
y = np.random.rand(10000)
print(X.shape, y.shape)
print("="*25)
# ----------------------------------------------------
# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print("="*25)
# ----------------------------------------------------
# Applying MiniBatchKMeans Model
MiniBatchKMeansModel = MiniBatchKMeans(n_clusters=5, batch_size=50, init='k-means++')  
MiniBatchKMeansModel.fit(X_train)
# ----------------------------------------------------
# Calculating Details
print('MiniBatchKMeansModel Train Score is : ', MiniBatchKMeansModel.score(X_train))
print('MiniBatchKMeansModel Test Score is : ', MiniBatchKMeansModel.score(X_test))
print('MiniBatchKMeansModel centers are : ', MiniBatchKMeansModel.cluster_centers_)
print('MiniBatchKMeansModel labels are : ', MiniBatchKMeansModel.labels_)
print('MiniBatchKMeansModel intertia is : ', MiniBatchKMeansModel.inertia_)
print('MiniBatchKMeansModel No. of iteration is : ', MiniBatchKMeansModel.n_iter_)
print("="*25)
# ----------------------------------------------------
# Calculating Prediction
y_pred = MiniBatchKMeansModel.predict(X_test)
print('Predicted Value for MiniBatchKMeansModel is : ', y_pred[:10])
print('True Value for MiniBatchKMeansModel is : ', y_test[:5])
# ----------------------------------------------------
plt.figure()
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, alpha=1);
plt.show(block=True) 
