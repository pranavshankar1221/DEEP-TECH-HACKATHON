

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os, glob

# ---------------------------- Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20           # Increase epochs for better learning
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMAGE_SIZE = 64           # Match your model input

data_dir = r"D:\DeepTech\data"
train_folder = os.path.join(data_dir, "train")
val_folder   = os.path.join(data_dir, "valid")
test_folder  = os.path.join(data_dir, "test")

# ---------------------------- Data Loaders (Grayscale + Augmentation)
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to 1 channel
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to 1 channel
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(train_folder, transform=train_transforms)
val_dataset   = datasets.ImageFolder(val_folder, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

class_names = train_dataset.classes
num_classes = len(class_names)

# ---------------------------- Define Custom CNN (1-channel input)
class MyCustomCNN(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(MyCustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 1 channel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)      # IMAGE_SIZE/4 = 16
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MyCustomCNN().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
best_val_acc = 0


for epoch in range(NUM_EPOCHS):
    model.train()
    correct, total, running_loss = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item()
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = 100 * correct / total
    val_loss /= len(val_loader)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "val_acc": val_acc
        }, "best_cnn_phase1.pth")

print("Training done. Best Val Acc:", best_val_acc)

# ---------------------------- Test Function
def predict_image(img_path, model, class_names):
    model.eval()
    img = Image.open(img_path).convert("L")  
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        max_prob, idx = torch.max(probs, 0)
    print(f"Image: {os.path.basename(img_path)} | "
          f"Predicted Class: {class_names[idx]} | Confidence: {max_prob.item()*100:.2f}%")


checkpoint = torch.load("best_cnn_phase1.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
for img_path in glob.glob(os.path.join(test_folder, "*.*")):
    predict_image(img_path, model, class_names)
