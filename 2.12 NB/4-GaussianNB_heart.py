from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# ----------------------------------------------------
dataset = pd.read_csv('path/2.12 NB/heart.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# ----------------------------------------------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# ----------------------------------------------------
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# ----------------------------------------------------
print('GaussianNBModel Train Score is : ' , classifier.score(X_train, y_train))
print('GaussianNBModel Test Score is : ' , classifier.score(X_test, y_test))
print("="*10)
y_pred = classifier.predict(X_test)
print('Pred Value for GaussianNBModel is : ', y_pred[:5])
print('True Value for GaussianNBModel is : ' , y_test[:5])
print("="*25)
# ----------------------------------------------------
CM = confusion_matrix(y_test, y_pred)
sns.heatmap(CM, center = True, annot=True, fmt="d")
plt.show()
# ----------------------------------------------------
