from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# ----------------------------------------------------
dataset = pd.read_csv('path/2.13 LDA QDA/iris.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# ---------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# ---------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# ---------------------------------------------------------------
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, y_train)
# ----------------------------
score1 = classifier1.score(X_test, y_test)
print('score 1 = ', score1 )
# ----------------------------
y_pred1 = classifier1.predict(X_test)
cm1 = confusion_matrix(y_test, y_pred1)
print('cm1 \n' , cm1 )
plt.figure()
sns.heatmap(cm1, center = True, annot=True, fmt="d")
plt.show(block=False) 
# ---------------------------------------------------------------
# apply LDA 
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
# ----------------------------
classifier2 = LogisticRegression(random_state = 0)
classifier2.fit(X_train, y_train)
# ----------------------------
score2 = classifier2.score(X_test, y_test)
print('score 2 = ', score2 )
# ----------------------------
y_pred2 = classifier2.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred2)
print('cm2 \n' , cm2 )
plt.figure()
sns.heatmap(cm2, center = True, annot=True, fmt="d")
plt.show(block=False) 
# ---------------------------------------------------------------

plt.show(block=True) 

