from sklearn.preprocessing import LabelEncoder
import pandas as pd
# ----------------------------------------------------
data = pd.read_csv('path/2.16 NLP (preprocessing)/mall.csv')
data.head()
df = pd.DataFrame(data)
print('Original dataframe is : \n', df)
print("="*25)
# ----------------------------------------------------
enc = LabelEncoder()
enc.fit(df['Genre'])
# ----------------------
print('classed found : ', list(enc.classes_))
print('equivilant numbers are : ', enc.transform(df['Genre']))
print("="*25)
# ----------------------
df['Genre Code'] = enc.transform(df['Genre'])
print('Updates dataframe is : \n', df)
print("="*25)
# ----------------------
print('Inverse Transform  : ', list(enc.inverse_transform([1, 0, 1, 1, 0, 0])))
print("="*25)
# ----------------------------------------------------
