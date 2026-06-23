import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ TRANSFORMS ------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  #  stronger augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------ DATASET ------------------

dataset = datasets.ImageFolder("data")  #  YOUR FOLDER

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_data, val_data = random_split(dataset, [train_size, val_size])

train_data.dataset.transform = train_transform
val_data.dataset.transform = val_transform

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# ------------------ MODEL ------------------

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# freeze most layers (very important fix)
for param in model.parameters():
    param.requires_grad = False

# unfreeze last layer only
for param in model.layer4.parameters():
    param.requires_grad = True

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.6),  #  stronger dropout
    nn.Linear(128, 2)
)

model = model.to(device)

# ------------------ LOSS ------------------

class_counts = [1081, 961]
weights = torch.tensor([1.0/class_counts[0], 1.0/class_counts[1]]).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)

# ------------------ OPTIMIZER ------------------

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=0.00003, weight_decay=1e-4)

# ------------------ TRAINING ------------------

epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    # -------- VALIDATION --------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {running_loss:.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Val Acc: {val_acc:.2f}%")

# ------------------ SAVE ------------------

import os

os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/deepfake_model.pth")
print(" Model saved successfully!")
# torch.save(model.state_dict(), "models/deepfake_model.pth")
# print(" Model saved successfully!")