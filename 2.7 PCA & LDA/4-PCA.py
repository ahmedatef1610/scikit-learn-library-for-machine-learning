import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# ----------------------------------------------------
rn = np.random.RandomState()
x = np.dot(rn.rand(2,2) , rn.randn(2,100) ).T
print(x.shape)
print("="*25)
# ----------------------------------------------------
model = PCA(n_components= 1)
model.fit(x)
data = model.transform(x)

print(x.shape)
print(data.shape)
print("="*25)
# ----------------------------------------------------
newdata = model.inverse_transform(data)
print(newdata.shape)
print("="*25)
# ----------------------------------------------------

plt.scatter(x[:,0],x[:,1],c="r")
plt.scatter(newdata[:,0],newdata[:,1],c="k")
plt.show()
# ----------------------------------------------------
