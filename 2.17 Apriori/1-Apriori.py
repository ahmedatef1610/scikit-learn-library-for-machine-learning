import pandas as pd  
# pip install apyori
from apyori import apriori
# ----------------------------------------------------
store_data = pd.read_csv('path/2.17 Apriori/store_data.csv', header=None)  
print(store_data.head())
print("="*25)
# ----------------------------------------------------
records = []  
# for i in range(0, 7501):  
for i in range(0, store_data.shape[0]):  
    records.append([str(store_data.values[i,j]) for j in range(0, 20)])

print(len(records))
print(records[5])
print("="*25)
# ----------------------------------------------------

association_rules = apriori(records, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)  
association_results = list(association_rules)  

print(len(association_results))  
print(association_results[0])  
'''
    RelationRecord(
        items=frozenset({'light cream', 'chicken'}), 
        support=0.004532728969470737, 
        ordered_statistics=[OrderedStatistic(items_base=frozenset({'light cream'}), items_add=frozenset({'chicken'}), confidence=0.29059829059829057, lift=4.84395061728395)]
    )
'''
print("="*25)
# ----------------------------------------------------

for item in association_results[:5] :

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    if len(items)>=2 :
        print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")

# ----------------------------------------------------
