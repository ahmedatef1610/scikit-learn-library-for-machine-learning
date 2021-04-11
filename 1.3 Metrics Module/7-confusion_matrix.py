from sklearn.metrics import confusion_matrix

# =======================================================================

y_true = ['a', 'b', 'b', 'a', 'b', 'a', 'a', 'b', 'a', 'b']
y_pred = ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'a', 'a', 'a']
CM = confusion_matrix(y_true, y_pred)
# tn, fp, fn, tp = CM.flatten()
# print(tn, fp, fn, tp)
print(CM)
print()
# =======================================================================

y_true = ['a', 'a', 'b', 'b', 'a', 'b', 'c', 'c', 'b', 'b']
y_pred = ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']
CM = confusion_matrix(y_true, y_pred)
print(CM)
print()

# =======================================================================

y_true = [9, 9, 8, 8, 5, 5, 9, 5, 8, 9, 8, 5]
y_pred = [5, 8, 9, 9, 8, 5, 5, 9, 8, 5, 9, 8]
CM = confusion_matrix(y_true, y_pred)
print(CM)
print()
