# Import Libraries
from sklearn.metrics import v_measure_score
# ----------------------------------------------------

print(v_measure_score([0, 0, 1, 1], [0, 0, 1, 1]))
print(v_measure_score([0, 0, 1, 1], [1, 1, 0, 0]))
print("="*25)
print(v_measure_score([0, 0, 1, 2], [0, 0, 1, 1]))
print(v_measure_score([0, 1, 2, 3], [0, 0, 1, 1]))
print("="*25)
print(v_measure_score([0, 0, 1, 1], [0, 0, 1, 2]))
print(v_measure_score([0, 0, 1, 1], [0, 1, 2, 3]))
print("="*25)
print(v_measure_score([0, 0, 0, 0], [0, 1, 2, 3]))
print("="*25)
print(v_measure_score([0, 0, 1, 1], [0, 0, 0, 0]))