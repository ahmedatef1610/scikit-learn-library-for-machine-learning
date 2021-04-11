# Import Libraries
from sklearn.metrics import completeness_score
# ----------------------------------------------------

print(completeness_score([0, 0, 1, 1], [1, 1, 0, 0]))
print("="*25)
print(completeness_score([0, 0, 1, 1], [0, 0, 0, 0]))
print(completeness_score([0, 1, 2, 3], [0, 0, 1, 1]))
print("="*25)
print(completeness_score([0, 0, 1, 1], [0, 1, 0, 1]))
print(completeness_score([0, 0, 0, 0], [0, 1, 2, 3]))