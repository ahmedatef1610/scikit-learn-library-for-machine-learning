import sklearn.svm as s
# import sklearn.externals.joblib as jb
import numpy as np
import joblib as jb
#----------------------------------------------------
x = np.random.randint(10,size =20).reshape(4,5)
y = [5,8,9,6]
#----------------------------------------------------
model = s.SVR()
model.fit(x,y)
#----------------------------------------------------
print(model.predict([[2,3,6,5,9]]))
jb.dump(model , 'path/3.4 Model Save/saved file.sav')
#----------------------------------------------------
savedmodel = jb.load('path/3.4 Model Save/saved file.sav')
print(savedmodel.predict([[2,3,6,5,9]]))

