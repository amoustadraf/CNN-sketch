# preview_data.py

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import matplotlib.pyplot as plt

# Load the saved dataset
data = np.load("./dataset/quickdraw_data.npz")
images = data["images"]
labels = data["labels"]

# Convert to PyTorch tensors
tensor_x = torch.Tensor(images)
tensor_y = torch.LongTensor(labels)

dataset = TensorDataset(tensor_x, tensor_y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Get one batch
dataiter = iter(loader)
images_batch, labels_batch = next(dataiter)

# Optional: define class labels (use same order as in prepare_data.py)
classes = [
    "airplane", "apple", "banana", "basketball", "bicycle", "book",
    "circle", "cloud", "diamond", "pizza", "smiley_face",
    "toilet", "triangle", "t-shirt"
]

# Make image grid
grid_img = torchvision.utils.make_grid(images_batch, nrow=8, padding=2, normalize=True)
npimg = grid_img.numpy()

# Plot the image grid
plt.figure(figsize=(12, 6))
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.title("Batch of training samples")
plt.axis("off")
plt.show()
