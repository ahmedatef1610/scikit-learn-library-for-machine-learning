#Import Libraries
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
#----------------------------------------------------
'''
sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None)

sklearn.metrics.plot_confusion_matrix(estimator, X, y_true, labels=None, sample_weight=None, normalize=None, display_labels=None, include_values=True, xticks_rotation='horizontal', values_format=None, cmap='viridis', ax=None, colorbar=True)

'''
y_true = ['a', 'b', 'b', 'a', 'b', 'a', 'a', 'b', 'a', 'b']
y_pred = ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'a', 'a', 'a']
#Calculating Confusion Matrix
CM = confusion_matrix(y_true, y_pred)
#print('Confusion Matrix is : ', CM)

# drawing confusion matrix
sns.heatmap(CM, center = True, annot=True, fmt="d")
plt.show()

#----------------------------------------------------

y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
CM = confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
sns.heatmap(CM, center = True, annot=True, fmt="d")
plt.show()

#----------------------------------------------------
import matplotlib.pyplot as plt  
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = SVC(random_state=0)
clf.fit(X_train, y_train)

plot_confusion_matrix(clf, X_test, y_test)  
plt.show()  


