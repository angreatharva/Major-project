import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import ImageFolder
from multiprocessing import freeze_support

# -------------------------------
# Squeeze-Excitation (SE) Block
# -------------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Global average pooling: shape (b, c)
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# -------------------------------
# Residual Block with SE
# -------------------------------
class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(ResidualSEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.se    = SEBlock(out_channels, reduction)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out

# -------------------------------
# ResEmoteNet Architecture
# -------------------------------
class ResEmoteNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ResEmoteNet, self).__init__()
        # Initial convolution: input size (B,3,224,224)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Four stages of residual blocks
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualSEBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualSEBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # (B,64,H/2,W/2)
        x = self.maxpool(x)                     # (B,64,H/4,W/4)
        x = self.layer1(x)                      # (B,64, ...)
        x = self.layer2(x)                      # (B,128, ...)
        x = self.layer3(x)                      # (B,256, ...)
        x = self.layer4(x)                      # (B,512, ...)
        x = self.avgpool(x)                     # (B,512,1,1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# -------------------------------
# Data Augmentation and DataLoaders
# -------------------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Adjust the 'dataset_path' below to your local RAF‑DB dataset location.
dataset_path = "DATASET"  # Folder with 'train' and 'test' subdirectories

train_dataset = ImageFolder(root=os.path.join(dataset_path, "train"), transform=train_transform)
test_dataset  = ImageFolder(root=os.path.join(dataset_path, "test"), transform=test_transform)

# For debugging, you can set num_workers=0
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# -------------------------------
# Training Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResEmoteNet(num_classes=7).to(device)
criterion = nn.CrossEntropyLoss()

# Using SGD with momentum and weight decay (tuned for high accuracy)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50)  # T_max equals number of epochs

# -------------------------------
# Training and Validation Functions
# -------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += torch.sum(preds == labels).item()
        total_samples += images.size(0)
        
        
            
    epoch_loss = running_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += images.size(0)
    val_loss = running_loss / total_samples
    val_acc = total_correct / total_samples
    return val_loss, val_acc

# -------------------------------
# Main Training Loop
# -------------------------------
if __name__ == '__main__':
    freeze_support()
    
    print("Starting training...")
    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))
    
    num_epochs = 50
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        scheduler.step()
        
        end_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs} completed in {end_time - start_time:.2f} seconds")
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_resemotenet_rafdb.pth")
    
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
