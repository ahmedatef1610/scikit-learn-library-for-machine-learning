import matplotlib.pyplot as plt
import pandas as pd
# ----------------------------------------------------
dataset = pd.read_csv('path/2.10 Ensemble Classifier/heart.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
# ----------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# ----------------------------------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# ----------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier.fit(X_train, y_train)
# ----------------------------------------------------
print('classifier Train Score is : ' , classifier.score(X_train, y_train))
print('classifier Test Score is : ' , classifier.score(X_test, y_test))
print("="*10)
# -------------------
print("the feature importances : \n", classifier.feature_importances_)
print("="*10)
# -------------------
y_pred = classifier.predict(X_test)
print(y_pred[:7])
print(y_test[:7])
print("="*10)
# ----------------------------------------------------
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import seaborn as sns
sns.heatmap(cm, center = True, annot=True, fmt="d")
plt.show()
# ----------------------------------------------------

for j in range(2,100):
   
    classifier = RandomForestClassifier(n_estimators = j, criterion = 'entropy')
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    print('RF for number of Trees : ' , j , ' is : \n' , cm)
    print('The Score is : ',classifier.score(X_test , y_test))
    print('=======================================================')