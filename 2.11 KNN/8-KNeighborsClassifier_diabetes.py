from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
# ----------------------------------------------------
df = pd.read_csv('path/2.11 KNN/diabetes.csv')
print(df.head())
print(df.shape)
# --------------------------
X = df.drop(columns=['Outcome'])
print(X.head())
y = df['Outcome'].values
print(y[:5])
print("="*25)
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
# ----------------------------------------------------
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)
# ----------------------------------------------------
print(knn.score(X_train, y_train))
print(knn.score(X_test, y_test))
print("="*10)
print(knn.predict(X_test)[:5])
print(y_test[:5])
print("="*25)
# ----------------------------------------------------
knn_cv = KNeighborsClassifier(n_neighbors=3)
cv_scores = cross_val_score(knn_cv, X, y, cv=5)
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))
# ----------------------------------------------------

# ------------------------
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 21 to good selection num neighbors
neighbors_settings = range(1, 40)
for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors,weights='uniform',algorithm = 'auto',p = 2,n_jobs=-1,)
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