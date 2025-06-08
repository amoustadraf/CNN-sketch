import torch, torch.optim as optim, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import SketchCNN
import numpy as np
from torch.utils.data import random_split # Importing necessary libraries. random_split is used to split the dataset into training and validation sets.
from torchvision import transforms # Importing transforms for data preprocessing.
from sklearn.metrics import classification_report, confusion_matrix

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # Randomly flips the image horizontally
    transforms.RandomRotation(10), # Randomly rotates the image by 10 degrees
    transforms.RandomResizedCrop(28, scale=(0.9, 1.0)), # Randomly crops the image to 28x28 pixels, with a scale between 80% and 100% of the original size
    transforms.ToTensor(), # Converts the image to a PyTorch tensor
])
# ^^This is a series of transformations that will be applied to the images in the dataset.

# Since my dataset is already in memory, not loaded via ImageFolder, so transforms won't work directly. Therefore, we create a custom dataset class to apply the transformations.
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # Convert the image to a PIL image for transformations
        img = transforms.ToPILImage()(img)  # Convert the tensor to a PIL image
        if self.transform:
            img = self.transform(img) # Apply the transformations
        return img, label
    


data = np.load("./dataset/quickdraw_data.npz")
images, labels = data["images"], data["labels"]
# PyTorch only works with its own data type called tensor (like a smart matrix)
##### tensor_x = torch.Tensor(images) # Convert images to float tensors
##### tensor_y = torch.LongTensor(labels) # Turns the labels into integers (for classification)

images_tensor = torch.Tensor(images)

labels_tensor = torch.LongTensor(labels)
dataset = AugmentedDataset(images_tensor, labels_tensor, transform=transform)

# We are adding validation set (80/20 split) to the dataset.
total_size = len(dataset) # Gets the total number of samples in the dataset (images).

train_size = int(0.8* total_size) # 80% of the dataset will be used for training.
val_size = total_size - train_size # The remaining 20% will be used for validation.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Separate into 2 loaders. 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) # Creates data loaders for the training and validation datasets.
# ^^Creates a data loader, which is an iterable over the dataset. It allows us to iterate over the dataset in batches, which is useful for training models. 
# The `shuffle=True` argument shuffles the dataset at every epoch, which helps with training. 
# Also, the `batch_size=32` argument specifies that we want to process 32 samples at a time. This is a common practice in deep learning to speed up training and reduce memory usage.

model = SketchCNN(num_classes=len(set(labels))) 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15
for epoch in range(epochs):
    running_loss = 0.0 # initializes the running loss for this epoch. At the end of the epoch, this will give us the average loss for the epoch.
    for inputs, labels in train_loader:
        optimizer.zero_grad() # resets the gradients to zero before each batch, so that we don't accumulate gradients from previous batches. Pytorch accumulates gradients by default, with the variable `grad` in the model's parameters.
        outputs = model(inputs)    # computes the model's predictions for the inputs
        loss = criterion(outputs, labels) # calculates the loss, i.e., how far off the predictions are from the actual labels. loss=(predictions - target/label)^2
        loss.backward() # calculates the gradient, aka backpropagation. Formula: gradient = d(loss)/d(weights) (derivative of loss with respect to weights)
        optimizer.step() # updates the model weights based on the gradients, formula: new_weight = old_weight - learning_rate * gradient
        running_loss += loss.item() # accumulates the loss for this batch. At the end of the epoch, this will give us the average loss for the epoch.
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}") # prints the average loss for the epoch

    model.eval() # Sets the model to evaluation mode. This is important because it changes the behavior of certain layers, like dropout and batch normalization, which behave differently during training and evaluation.
    val_loss =0.0
    correct = 0
    total = 0
    with torch.no_grad(): # Disables gradient calculation, which is not needed during evaluation and saves memory. Especially since we're not updating the model weights during validation!!!!
        for inputs, labels in val_loader:
            outputs = model(inputs) # Computes the model's predictions for the validation inputs.
            loss = criterion(outputs, labels) # Calculates the loss for the validation set.
            val_loss+= loss.item() # Accumulates the validation loss.

            preds = torch.argmax(outputs, dim=1) # Which answer did the model pick for each image? Model sees an image and gives 5 scores (like guessing 5 multiple-choice answers). argmax picks the biggest score.
            # ^^Gets the predicted labels by taking the index of the maximum value in the output tensor along dimension 1 (the class dimension).
            correct += (preds == labels).sum().item() # How many of those guesses were right? It checks "did the guess match the correct label?" and adds the number of right answers to your total correct answers.
            # ^^Counts the number of correct predictions by comparing the predicted labels with the actual labels.
            total += labels.size(0) # How many total guesses did we make? Just counts how many images we teested in this batch.
            # ^^Counts the total number of samples in the validation set.
    val_acc = correct / total
    # Let's say your model guessed preds = [1, 0, 2] and labels = [1, 2, 2]. This means it got 2 correct out of 3 (first and last). So correct += 2 and total += 3. Then val_acc = 2 / 3 = 0.6667 (or 66.67% accuracy).
    print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
    model.train() # Sets the model back to training mode for the next epoch.
torch.save(model.state_dict(), "sketch_cnn.pth") # Saves the model's state dictionary (i.e., the model's parameters) to a file called "sketch_cnn.pth". This allows us to load the model later without having to retrain it.
print("Training done!")
# === Classification report & confusion matrix ===
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, targets in val_loader:
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(targets.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Define class names in the same order
classes = [
    "Airplane", "Apple", "Basketball", "Bicycle", "Book",
    "Circle", "Cloud", "Diamond", "Pizza", "Smiley Face",
    "Toilet", "Triangle", "T-Shirt"
]

print("=== CLASSIFICATION REPORT ===")
print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

print("=== CONFUSION MATRIX ===")
print(confusion_matrix(all_labels, all_preds))