import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  

# === Load your data ===
data = np.load("tsne_output.npz")
coords = data["coords"]
labels = data["labels"]

class_names = [
    "Airplane", "Apple", "Basketball", "Bicycle", "Book",
    "Circle", "Cloud", "Diamond", "Pizza", "Smiley Face",
    "Toilet", "Triangle", "T-Shirt"
]

# Create DataFrame
df = pd.DataFrame(coords, columns=["x", "y", "z"])
df["label"] = labels

# === COLORS ===
num_classes = len(class_names)
cmap = cm.get_cmap('tab20', num_classes)  
colors = [cmap(i) for i in range(num_classes)]

# === 1. 3D t-SNE Plot ===
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
for i in range(num_classes):
    subset = df[df["label"] == i]
    ax.scatter(subset["x"], subset["y"], subset["z"], color=colors[i], label=class_names[i], s=10)
ax.legend(title="Class", loc='upper left', bbox_to_anchor=(1.15, 1))
ax.set_title("3D t-SNE of Sketch Dataset")
plt.tight_layout()
plt.show()

# === 2. 2D t-SNE Plot ===
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
X_2d = tsne_2d.fit_transform(df[["x", "y", "z"]].values)
df["x_2d"] = X_2d[:, 0]
df["y_2d"] = X_2d[:, 1]

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    subset = df[df["label"] == i]
    plt.scatter(subset["x_2d"], subset["y_2d"], s=10, label=class_names[i], color=colors[i])
plt.title("2D t-SNE of Sketch Dataset")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Class", loc='upper right', bbox_to_anchor=(1.25, 1))
plt.tight_layout()
plt.show()

# === 3. Distance Between Cluster Centers ===
centers = df.groupby("label")[["x", "y", "z"]].mean().to_numpy()
dist_matrix = pairwise_distances(centers)

plt.figure(figsize=(10, 8))
sns.heatmap(dist_matrix, annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names, cmap="coolwarm")
plt.title("Distance Between Class Clusters (t-SNE Space)")
plt.tight_layout()
plt.show()