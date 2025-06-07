import torch, torch.optim as optim, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import SketchCNN
import numpy as np

data = np.load("./dataset/quickdraw_data.npz")
images, labels = data["images"], data["labels"]

tensor_x = torch.Tensor(images)
tensor_y = torch.LongTensor(labels)

dataset = TensorDataset(tensor_x, tensor_y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SketchCNN(num_classes=len(set(labels)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(loader):.4f}")

torch.save(model.state_dict(), "sketch_cnn.pth")
print("✅ Training done!")