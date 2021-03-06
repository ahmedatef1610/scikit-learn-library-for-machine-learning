from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# ----------------------------------------------------
data = load_breast_cancer()
X = data.data
y = data.target
# ----------------------------------------------------
poly = PolynomialFeatures( degree = 3 , include_bias = False)
poly.fit(X)
x_poly = poly.transform(X)
print(x_poly.shape)
print("="*25)
# ----------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size = 0.2)
# ----------------------------------------------------
# apply LR
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train , y_train)

result= logreg.predict(x_test)
print('accuracy =',accuracy_score(y_test , result))
 
cm = confusion_matrix(y_test , result)
print('confusion matrix \n',  cm)


import seaborn as sns
sns.heatmap(cm, center=True, annot=True, fmt="d")
plt.show()
print("="*25)
# ----------------------------------------------------
# apply SVC
from sklearn.svm import SVC

classifier = SVC()
classifier.fit(x_train , y_train)

result= logreg.predict(x_test)
print('accuracy =',accuracy_score(y_test , result))
 
cm = confusion_matrix(y_test , result)
print('confusion matrix \n',  cm)


sns.heatmap(cm, center=True, annot=True, fmt="d")
plt.show()
# ----------------------------------------------------



