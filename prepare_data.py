# prepare_data.py

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def draw_sketch(strokes, size=28, lw=4):
    large_size = 256
    img = Image.new("L", (large_size, large_size), 255)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        points = list(zip(stroke[0], stroke[1]))
        draw.line(points, fill=0, width=lw)
    return img.resize((size, size), resample=Image.LANCZOS)

# Filenames for each class
raw_filenames = [
    "full_simplified_airplane", "full_simplified_apple", "full_simplified_basketball",
    "full_simplified_bicycle", "full_simplified_book", "full_simplified_circle",
    "full_simplified_cloud", "full_simplified_diamond", "full_simplified_pizza",
    "full_simplified_smiley face", "full_simplified_toilet",
    "full_simplified_triangle", "full_simplified_t-shirt"
]

data_path = "./data/"
save_path = "./dataset/"
os.makedirs(save_path, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

images, labels = [], []

for idx, cls in enumerate(raw_filenames):
    filepath = os.path.join(data_path, f"{cls}.ndjson")
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            sample = json.loads(line)
            img = draw_sketch(sample["drawing"])
            img_tensor = transform(img)  # shape (1,28,28), float tensor
            images.append(img_tensor)
            labels.append(idx)

# Visualize a few samples to sanity-check
for i in range(5):
    img = images[i][0]  # take the first channel
    label = labels[i]
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis("off")
plt.show()

# Stack tensors and save
images_tensor = torch.stack(images)                  # shape (N,1,28,28)
labels_tensor = torch.tensor(labels, dtype=torch.int64)

np.savez(
    os.path.join(save_path, "quickdraw_data.npz"),
    images=images_tensor.numpy(),
    labels=labels_tensor.numpy()
)
print("✅ Data saved!")
