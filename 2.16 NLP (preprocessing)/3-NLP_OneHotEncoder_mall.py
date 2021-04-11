from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
# ----------------------------------------------------
data = pd.read_csv('path/2.16 NLP (preprocessing)/mall.csv')
print(data.head())
print("="*25)
df = pd.DataFrame(data)
print('Original dataframe is : \n', df)
print("="*25)
# ----------------------------------------------------
ohe = OneHotEncoder()
col = np.array(df['Genre'])
col = col.reshape(len(col), 1)
ohe.fit(col)
newmatrix = ohe.transform(col).toarray()
newmatrix = newmatrix.T
print(newmatrix)
print("="*25)
# ----------------------
df['Female'] = newmatrix[0]
df['male'] = newmatrix[1]

print('Updates dataframe is : \n', df)
# ----------------------------------------------------
