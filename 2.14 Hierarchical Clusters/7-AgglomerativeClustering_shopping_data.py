import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt  
import pandas as pd  
# ----------------------------------------------------
customer_data = pd.read_csv('path/2.14 Hierarchical Clusters/shopping_data.csv')  
# ----------
print(customer_data.shape)
print(customer_data.head())
# ----------
data = customer_data.iloc[:, 3:5].values  
print(data.shape)
print("="*25)
# ----------------------------------------------------
plt.figure(figsize=(8, 6))  
plt.title("Customer Dendograms")  
dendrogram = shc.dendrogram(shc.linkage(data, method='ward')) 
# ----------------------------------------------------
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data) 
# ----------------------------------------------------
print("="*25)