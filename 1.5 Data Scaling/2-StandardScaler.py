from sklearn.preprocessing import StandardScaler

data = [[5, 455], [555, 1000], [545, 4568], [12354,6305]]

scaler = StandardScaler()

scaler.fit(data)
print(scaler.mean_)

newdata = scaler.transform(data)
print(newdata)

newdata = scaler.fit_transform(data)
print(newdata)
