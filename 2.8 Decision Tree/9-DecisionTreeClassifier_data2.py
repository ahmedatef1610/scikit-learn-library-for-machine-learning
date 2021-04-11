import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# ----------------------------------------------------
balance_data = pd.read_csv('path/2.8 Decision Tree/data2.csv',header=None)
print ("Dataset Lenght : ", len(balance_data))
print ("Dataset Shape : ", balance_data.shape)
print ("Dataset : \n", balance_data.head())

print("="*25)
print(balance_data[0].value_counts())
print("="*25)
# dataset.iloc[:, 0:1].values
X = balance_data.values[:, 1:]
Y = balance_data.values[:,0]
print(X[:7])
print(Y[:7])
print("="*25)
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
# ----------------------------------------------------
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
# ----------------------------------------------------
print('clf_gini Train Score is : ', clf_gini.score(X_train, y_train))
print('clf_gini Test Score is : ', clf_gini.score(X_test, y_test))
print("="*10)
print('clf_entropy Train Score is : ', clf_entropy.score(X_train, y_train))
print('clf_entropy Test Score is : ', clf_entropy.score(X_test, y_test))
print("="*25)
print("the feature importances : \n", clf_gini.feature_importances_)
print("="*10)
print("the feature importances : \n", clf_entropy.feature_importances_)
print("="*25)
# ----------------------------------------------------
print(clf_gini.predict([[4, 4, 3, 3]]))
print(clf_entropy.predict([[4, 4, 3, 3]]))
print("="*10)

y_pred_gini = clf_gini.predict(X_test)
print(y_pred_gini[:7])
print(Y[:7])
print("="*10)

y_pred_entropy = clf_entropy.predict(X_test)
print(y_pred_entropy[:7])
print(Y[:7])
print("="*10)

print ("Accuracy for gini is ", accuracy_score(y_test,y_pred_gini)*100)
print ("Accuracy for entropy is ", accuracy_score(y_test,y_pred_entropy)*100)
print("="*25)
# ----------------------------------------------------

    