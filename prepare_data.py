import os, json, numpy as np
from PIL import Image, ImageDraw
import torch, torchvision.transforms as transforms

def draw_sketch(strokes, size=28, lw=4):
    large_size = 256
    img = Image.new("L", (large_size, large_size), 255)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        points = list(zip(stroke[0], stroke[1]))
        draw.line(points, fill=0, width=lw)
    return img.resize((size, size), resample=Image.LANCZOS)

classes=["full_simplified_airplane",'full_simplified_apple','full_simplified_banana','full_simplified_basketball','full_simplified_bicycle','full_simplified_book','full_simplified_circle','full_simplified_cloud','full_simplified_diamond','full_simplified_pizza','full_simplified_smiley face','full_simplified_toilet','full_simplified_triangle','full_simplified_t-shirt']
data_path = "./data/"
save_path = "./dataset/"
os.makedirs(save_path, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

images, labels = [], []

for idx, cls in enumerate(classes):
    with open(f"{data_path}/{cls}.ndjson") as f:
        for i, line in enumerate(f):
            if i == 1000: break  # limit to 1000 sketches per class
            sample = json.loads(line)
            img = draw_sketch(sample["drawing"])
            img_tensor = transform(img)
            images.append(img_tensor.numpy())
            labels.append(idx)

images = np.array(images)
labels = np.array(labels)

import matplotlib.pyplot as plt

# Pick a few examples to visualize
for i in range(5):
    img = images[i][0]  # images[i] is (1, 28, 28), we take the first channel
    label = labels[i]
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()

np.savez(f"{save_path}/quickdraw_data.npz", images=images, labels=labels)
print("✅ Data saved!")