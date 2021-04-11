from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile ,f_classif , chi2 
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------------------
data = load_breast_cancer()
X = data.data
y = data.target
print(data.feature_names.__len__())
# -------------------------------------------------------------------
# sel = SelectPercentile(score_func = chi2 , percentile = 20)
# sel = SelectPercentile(score_func = chi2 , percentile = 40)
# sel = SelectPercentile(score_func = chi2 , percentile = 60)
# sel = SelectPercentile(score_func = chi2, percentile = 80)

sel = SelectPercentile(score_func = f_classif , percentile = 5)
sel.fit(X,y)
selected_features = sel.transform(X)
print(selected_features)
print('='*25)

sFeatures = sel.get_support()
print('Selected features = \n' , sFeatures)
print(data.feature_names[sel.get_support()])
print('='*25)

scale  = MinMaxScaler()
scale.fit(selected_features)
selected_features = scale.transform(selected_features)
# -------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(selected_features, y, test_size = 0.2)
# -------------------------------------------------------------------
logreg = LogisticRegression()
logreg.fit(x_train , y_train)
result= logreg.predict(x_test)

print('accuracy = ',accuracy_score(y_test , result))
print('='*25)

conf = confusion_matrix(y_test , result)
print('confusion matrix \n',  conf)
print('='*25)
# ----------------------------------------------------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

x_axis = np.arange(0-0.1, 1+0.1, 0.001)
xx0, xx1 = np.meshgrid(x_axis,x_axis)
Z = logreg.predict(np.c_[xx0.ravel(), xx1.ravel()]).reshape(xx0.shape)

sns.scatterplot(x=selected_features[:,0], y=selected_features[:,1], hue=y, alpha=1);
plt.contourf(xx0, xx1, Z, alpha=0.2, cmap=plt.cm.Paired)
plt.show()

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, result)
sns.heatmap(CM, center = True, annot=True, fmt="d")
plt.show()
# ----------------------------