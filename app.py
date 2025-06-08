# app.py

import streamlit as st
import torch
import torchvision.transforms as transforms
from model import SketchCNN
from PIL import Image, ImageOps
import numpy as np

# 13 classes in the correct order
classes = [
    "Airplane", "Apple", "Basketball", "Bicycle", "Book",
    "Circle", "Cloud", "Diamond", "Pizza", "Smiley Face",
    "Toilet", "Triangle", "T-Shirt"
]

# Load model
model = SketchCNN(num_classes=len(classes))
model.load_state_dict(torch.load("sketch_cnn.pth"))
model.eval()

# Function to crop & center the sketch
def crop_and_center(pil_img):
    arr = np.array(pil_img)
    mask = arr < 255
    coords = np.argwhere(mask)
    if coords.size == 0:
        return pil_img.resize((28, 28), Image.LANCZOS)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = pil_img.crop((x0, y0, x1, y1))
    w, h = cropped.size
    pad = abs(h - w) // 2
    if w > h:
        cropped = ImageOps.expand(cropped, border=(0, pad, 0, pad), fill=255)
    else:
        cropped = ImageOps.expand(cropped, border=(pad, 0, pad, 0), fill=255)
    return cropped.resize((28, 28), Image.LANCZOS)

# Base transform (match training: grayscale [0–1])
transform = transforms.Compose([
    transforms.ToTensor(),
])

st.title("Sketch-to-Emoji")

uploaded = st.file_uploader("Upload your sketch (black lines on white background)", type=["png", "jpg", "jpeg"])
if uploaded:
    # Load, invert, crop, center, and display
    img = Image.open(uploaded).convert("L")
    img = crop_and_center(img)
    st.image(img, caption="Cropped & Centered Sketch", width=200)

    # Test-time augmentation: rotate and average probabilities
    angles = [0, 90, 180, 270]
    probs_sum = None

    for a in angles:
        img_rot = img.rotate(a)
        tensor = transform(img_rot).unsqueeze(0)  # shape [1,1,28,28]
        with torch.no_grad():
            p = torch.softmax(model(tensor), dim=1)
        probs_sum = p if probs_sum is None else probs_sum + p

    avg_probs = probs_sum / len(angles)
    top3 = torch.topk(avg_probs, 3)

    # Show results
    st.subheader("Top 3 Predictions:")
    for i in range(3):
        idx = top3.indices[0][i].item()
        conf = top3.values[0][i].item() * 100
        st.write(f"{classes[idx]} — {conf:.2f}%")
