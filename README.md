# 🖌️ Sketch CNN Classifier

This project is a Convolutional Neural Network (CNN) built **from scratch** to recognize hand-drawn sketches using Google's [QuickDraw dataset](https://github.com/googlecreativelab/quickdraw-dataset).

---

## 🚀 Features

- Pure PyTorch implementation 
- Data augmentation (rotation, scale, flip, etc.)
- t-SNE visualizations for embedding clusters (run the preview_data.py file)
- Real-time sketch testing via **Streamlit app** (streamlit run app.py)
- Top-3 class predictions
- Adam optimizer, dropout, and batch normalization

---

## 🧠 Model Architecture
Input (1x28x28 grayscale)
→ Conv2D(32) + ReLU + BatchNorm
→ Conv2D(64) + ReLU + BatchNorm + MaxPool
→ Conv2D(128) + ReLU + Dropout
→ Flatten → Dense(256) → ReLU → Dropout
→ Output Layer (Softmax over N classes)

---

## 🧪 To Run Locally

```bash
git clone https://github.com/amoustadraf/CNN-sketch
cd CNN-sketch  # <- or whatever folder name you chose when cloning
pip install -r requirements.txt
streamlit run app/app.py
