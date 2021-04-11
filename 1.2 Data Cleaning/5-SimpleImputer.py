from sklearn.impute import SimpleImputer


data = [[1,2,0],
        [3,0,1],
        [5,0,0],
        [0,4,6],
        [5,0,0],
        [4,5,5]]


imputer = SimpleImputer(missing_values=0, strategy='mean')
imputer = imputer.fit(data)
modifieddata = imputer.transform(data)
print(modifieddata)


# modifieddata = imputer.fit_transform(data)
# print(modifieddata)