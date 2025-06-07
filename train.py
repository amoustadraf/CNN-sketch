import torch, torch.optim as optim, torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from model import SketchCNN
import numpy as np

# Load and normalize
data = np.load("./dataset/quickdraw_data.npz")
images = data["images"].astype(np.float32) / 255.0
labels = data["labels"]
images = np.expand_dims(images, axis=1)  # (N, 1, 28, 28)

# Create dataset
tensor_x = torch.Tensor(images)
tensor_y = torch.LongTensor(labels)
full_dataset = TensorDataset(tensor_x, tensor_y)

# Split dataset
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model setup
model = SketchCNN(num_classes=len(set(labels)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation
epochs = 15
for epoch in range(epochs):
    model.train()
    train_loss, correct = 0.0, 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = torch.argmax(outputs, dim=1)
        correct += (pred == targets).sum().item()
    train_acc = correct / len(train_dataset)

    # Validation
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            outputs = model(val_inputs)
            loss = criterion(outputs, val_targets)
            val_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            val_correct += (pred == val_targets).sum().item()
    val_acc = val_correct / len(val_dataset)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f}")
