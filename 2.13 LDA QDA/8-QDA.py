import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#----------------------------------------------------
X = np.random.randint(20, size=(50, 10)) 
y = np.random.randint(5, size=(50, 1)).flatten()
#----------------------------------------------------
Qclf = QDA()
Qclf.fit(X, y)
#------------
QDAScore = Qclf.score(X,y)
print('Quadratic Score = ' , QDAScore )
print("="*10)
#------------
z = np.random.randint(20, size=(1, 10))
print('Quadratic Prediction = ' , Qclf.predict(z))
print("="*25)
#----------------------------------------------------
Lclf = LDA()
Lclf.fit(X, y)
#------------
LDAScore = Lclf.score(X,y)
print('Linear Score = ' , LDAScore )
print("="*10)
#------------
z = np.random.randint(20, size=(1, 10))
print('Linear Prediction = ' , Lclf.predict(z))
print("="*25)
#----------------------------------------------------

