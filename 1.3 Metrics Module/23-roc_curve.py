import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])

fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
print('fpr Value  : ', fpr)
print('tpr Value  : ', tpr)
print('thresholds Value  : ', thresholds)

print()
print()
print()

fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
plt.plot(fpr, tpr)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()