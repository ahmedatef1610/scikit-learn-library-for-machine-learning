# Support Vector Machine (SVM)
# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
# ----------------------------------------------------
# Importing the dataset
dataset = pd.read_csv('path/2.5 SVC/heart.csv')
X = dataset.iloc[:,:-1]
y = dataset.iloc[:, -1]
# ----------------------------------------------------
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# ----------------------------------------------------
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# ----------------------------------------------------
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
# ----------------------------------------------------
print('classifier Train Score is : ' , classifier.score(X_train, y_train))
print('classifier Test Score is : ' , classifier.score(X_test, y_test))
print("="*25)
# ----------------------------------------------------
# Predicting the Test set results
y_pred = classifier.predict(X_test)
print('Pred Value for classifier is : ' , y_pred[:5])
print('True Value for classifier is : ' , y_test[:5])
print("="*25)
# ----------------------------------------------------
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# ----------------------------------------------------
import seaborn as sns
sns.heatmap(cm, center=True, annot=True, fmt="d")
plt.show()

  