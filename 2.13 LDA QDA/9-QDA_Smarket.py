import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
#----------------------------------------------------
df = pd.read_csv('path/2.13 LDA QDA/Smarket.csv', index_col=0, parse_dates=True)
print(df.head())
print("="*25)
#--------------------
X_train = df[:'2004'][['Lag1','Lag2']]
y_train = df[:'2004']['Direction']

X_test = df['2005':][['Lag1','Lag2']]
y_test = df['2005':]['Direction']
#----------------------------------------------------
lda = LinearDiscriminantAnalysis()
model = lda.fit(X_train, y_train)

pred=model.predict(X_test)
print('Prediction for LDA : ', np.unique(pred, return_counts=True))
print('CM for LDA : \n', confusion_matrix(pred, y_test))
print('Report for LDA \n: ', classification_report(y_test, pred, digits=3))
print("="*25)
#----------------------------------------------------
qda = QuadraticDiscriminantAnalysis()
model2 = qda.fit(X_train, y_train)

pred2=model2.predict(X_test)
print('Prediction for QDA : ', np.unique(pred2, return_counts=True))
print('CM for QDA : \n', confusion_matrix(pred2, y_test))
print('Report for QDA : \n', classification_report(y_test, pred2, digits=3))
print("="*25)

#----------------------------------------------------


