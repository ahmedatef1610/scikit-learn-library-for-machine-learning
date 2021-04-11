import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# ----------------------------------------------------
dataset = pd.read_csv('path/2.11 KNN/houses.csv')
print(dataset.head())
print("="*25)
# ------------------
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(dataset)
dataset = imp.transform(dataset)
# ------------------
X = dataset[:, :-1]
y = dataset[:, -1]
print(X.shape)
print(y.shape)
print("="*25)
# ------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
# ------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print("="*25)
# ----------------------------------------------------
from sklearn import neighbors
knn = neighbors.KNeighborsRegressor(n_neighbors=4, weights='uniform', algorithm = 'auto', p = 2, n_jobs=-1,)
knn.fit(X_train, y_train)
# ---------------
print('KNeighborsRegressorModel Train Score is : ' , knn.score(X_train, y_train))
print('KNeighborsRegressorModel Test Score is : ' , knn.score(X_test, y_test))
print("="*10)
# ---------------
y_pred = knn.predict(X_test) 
print(y_pred[:5])
print(y_test[:5])
print("="*25)
# ----------------------------------------------------
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)
# ---------------
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
# ----------------------------------------------------
# ------------------------
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 21 to good selection num neighbors
neighbors_settings = range(1, 40)
for n_neighbors in neighbors_settings:
    # build the model
    knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors,weights='uniform',algorithm = 'auto',p = 2,n_jobs=-1,)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
plt.figure('choose', figsize=(10,5))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.xticks(np.arange(0,40,1))
plt.legend()
plt.tight_layout()
plt.show(block=False)

# ------------------------
plt.show(block=True) 