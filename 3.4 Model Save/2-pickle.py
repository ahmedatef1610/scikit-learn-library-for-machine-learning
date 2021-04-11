import sklearn.svm as s
import pickle as pk
import numpy as np
#----------------------------------------------------
x = np.random.randint(10,size =20).reshape(4,5)
y = [5,8,9,6]
#----------------------------------------------------
model = s.SVR()
model.fit(x,y)
#----------------------------------------------------
print(model.predict([[2,3,6,5,9]]))
pk.dump(model , open('path/3.4 Model Save/saved file2.sav','wb'))
#----------------------------------------------------
savedmodel = pk.load(open('path/3.4 Model Save/saved file2.sav','rb'))
print(savedmodel.predict([[2,3,6,5,9]]))
