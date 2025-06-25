import numpy as np

y = np.load("y_mri_balanced.npy")
unique, counts = np.unique(y, return_counts=True)
class_distribution = dict(zip(unique, counts))
print(class_distribution)
