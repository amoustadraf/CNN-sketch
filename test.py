import numpy as np
import matplotlib.pyplot as plt

data = np.load("dataset/quickdraw_data.npz")
images, labels = data["images"], data["labels"]

# Airplane class = 0 (check if different in your list)
airplane_imgs = images[labels == 0][:9]  # first 9 airplane samples

plt.figure(figsize=(6, 6))
for i, img in enumerate(airplane_imgs):
    plt.subplot(3, 3, i + 1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.axis('off')
plt.suptitle("Airplane Samples")
plt.tight_layout()
plt.show()

import numpy as np

data = np.load("./dataset/quickdraw_data.npz")
images = data["images"]
labels = data["labels"]

import numpy as np

unique_labels, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Label {label}: {count} samples")
classes = [
    "Airplane", "Apple", "Basketball", "Bicycle", "Book",
    "Circle", "Cloud", "Diamond", "Pizza", "Smiley Face", "Toilet",
    "Triangle", "T-Shirt"
]
for label, count in zip(unique_labels, counts):
    print(f"{classes[label]} (Label {label}): {count} samples")
