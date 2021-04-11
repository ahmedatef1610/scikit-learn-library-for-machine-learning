from sklearn.preprocessing import MinMaxScaler

data = [[-1, 2], 
        [-0.5, 6], 
        [0, 10], 
        [1, 18]]

scaler = MinMaxScaler(copy=True)

scaler.fit(data)
print(scaler.data_range_)
print()
print(scaler.data_min_)
print()
print(scaler.data_max_)
print()

newdata = scaler.transform(data)
print(newdata)
print()

scaler = MinMaxScaler(copy=True, feature_range=(1, 5))
newdata = scaler.fit_transform(data)
print(newdata)
