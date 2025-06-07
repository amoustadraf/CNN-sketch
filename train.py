import torch, torch.optim as optim, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import SketchCNN
import numpy as np

data = np.load("./dataset/quickdraw_data.npz")
images, labels = data["images"], data["labels"]
# PyTorch only works with its own data type called tensor (like a smart matrix)
tensor_x = torch.Tensor(images) # Convert images to float tensors
tensor_y = torch.LongTensor(labels) # Turns the labels into integers (for classification)

dataset = TensorDataset(tensor_x, tensor_y) # Creates a dataset from the tensors. This is a PyTorch dataset, which is a collection of data samples and their corresponding labels.
loader = DataLoader(dataset, batch_size=32, shuffle=True) 
# ^^Creates a data loader, which is an iterable over the dataset. It allows us to iterate over the dataset in batches, which is useful for training models. 
# The `shuffle=True` argument shuffles the dataset at every epoch, which helps with training. 
# Also, the `batch_size=32` argument specifies that we want to process 32 samples at a time. This is a common practice in deep learning to speed up training and reduce memory usage.

model = SketchCNN(num_classes=len(set(labels))) 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15
for epoch in range(epochs):
    running_loss = 0.0 # initializes the running loss for this epoch. At the end of the epoch, this will give us the average loss for the epoch.
    for inputs, labels in loader:
        optimizer.zero_grad() # resets the gradients to zero before each batch, so that we don't accumulate gradients from previous batches. Pytorch accumulates gradients by default, with the variable `grad` in the model's parameters.
        outputs = model(inputs)    # computes the model's predictions for the inputs
        loss = criterion(outputs, labels) # calculates the loss, i.e., how far off the predictions are from the actual labels. loss=(predictions - target/label)^2
        loss.backward() # calculates the gradient, aka backpropagation. Formula: gradient = d(loss)/d(weights) (derivative of loss with respect to weights)
        optimizer.step() # updates the model weights based on the gradients, formula: new_weight = old_weight - learning_rate * gradient
        running_loss += loss.item() # accumulates the loss for this batch. At the end of the epoch, this will give us the average loss for the epoch.
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(loader):.4f}") # prints the average loss for the epoch

torch.save(model.state_dict(), "sketch_cnn.pth") # Saves the model's state dictionary (i.e., the model's parameters) to a file called "sketch_cnn.pth". This allows us to load the model later without having to retrain it.
print("Training done!")