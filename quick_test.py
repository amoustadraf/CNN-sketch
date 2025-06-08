# quick_test.py

import torch
import numpy as np
from model import SketchCNN
from torchvision import transforms
from PIL import Image

# 1) Define your classes in the same order you trained
classes = [
    "Airplane", "Apple", "Basketball", "Bicycle", "Book",
    "Circle", "Cloud", "Diamond", "Pizza", "Smiley Face",
    "Toilet", "Triangle", "T-Shirt"
]

# 2) Load your model
model = SketchCNN(num_classes=len(classes))
model.load_state_dict(torch.load("sketch_cnn.pth"))
model.eval()

# 3) Grab a known-airplane from your .npz
data = np.load("dataset/quickdraw_data.npz")
images, labels = data["images"], data["labels"]
img_arr = images[labels == 0][0]  # first sample where label==0 (Airplane)
img = Image.fromarray((img_arr[0] * 255).astype(np.uint8), mode="L")

# 4) Preprocess exactly as in your app
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])
tensor = transform(img).unsqueeze(0)  # shape [1,1,28,28]

# 5) Inference
with torch.no_grad():
    pred = model(tensor)
    top = torch.argmax(pred, dim=1).item()

print(
    f"True label = 0 (Airplane). "
    f"Model predicted = {top} → “{classes[top]}”"
)
