# app.py
import streamlit as st
import torch
import torchvision.transforms as transforms
from model import SketchCNN
from PIL import Image
import numpy as np

# Updated class list to match 14 categories
classes = [
    "✈️ Airplane", "🍎 Apple", "🍌 Banana", "🏀 Basketball", "🚲 Bicycle", "📖 Book",
    "⚪ Circle", "☁️ Cloud", "💎 Diamond", "🍕 Pizza", "😊 Smiley Face", "🚽 Toilet",
    "🔺 Triangle", "👕 T-Shirt"
]

# Load model
model = SketchCNN(num_classes=len(classes))
model.load_state_dict(torch.load("sketch_cnn.pth"))
model.eval()

# Transform image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# Streamlit UI
st.title("Sketch-to-Emoji")

uploaded = st.file_uploader("Upload your sketch (black lines on white background)", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert('L')
    st.image(img, caption="Your sketch", width=200)

    tensor = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        pred = model(tensor)
        probs = torch.softmax(pred, dim=1)
        confidence, predicted = torch.max(probs, 1)

    st.header(f"Prediction: {classes[predicted]} ({confidence.item()*100:.2f}%)")