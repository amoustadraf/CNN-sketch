import numpy as np
data = np.load("./dataset/quickdraw_data.npz")
labels = data["labels"]
classes, counts = np.unique(labels, return_counts=True)
print(dict(zip(classes, counts)))