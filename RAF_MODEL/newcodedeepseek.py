import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.datasets import ImageFolder
from multiprocessing import freeze_support
import numpy as np

# -------------------------------
# Fixed Mixup Implementation
# -------------------------------
class Mixup:
    def __init__(self, alpha=0.4, prob=0.5, num_classes=7):
        self.alpha = alpha
        self.prob = prob
        self.num_classes = num_classes
        
    def __call__(self, inputs, targets):
        # Always convert to one-hot first
        targets_onehot = F.one_hot(targets, self.num_classes).float()
        
        if np.random.rand() > self.prob:
            return inputs, targets_onehot
            
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = inputs.size(0)
        index = torch.randperm(batch_size).to(inputs.device)
        
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        mixed_targets = lam * targets_onehot + (1 - lam) * targets_onehot[index]
        
        return mixed_inputs, mixed_targets

# -------------------------------
# Proper Loss Function
# -------------------------------
class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, preds, targets):
        log_probs = F.log_softmax(preds, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1)
        return loss.mean()

# -------------------------------
# Network Architecture (Unchanged)
# -------------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(ResidualSEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction)
        self.dropout = nn.Dropout2d(0.1)
        self.act = nn.SiLU(inplace=True)
        
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
            
        x += identity
        return self.act(x)

class ResEmoteNetV2(nn.Module):
    def __init__(self, num_classes=7):
        super(ResEmoteNetV2, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.layers = nn.Sequential(
            self._make_layer(128, 128, 3, 1),
            self._make_layer(128, 256, 3, 2),
            self._make_layer(256, 512, 4, 2),
            self._make_layer(512, 1024, 4, 2),
        )
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResidualSEBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualSEBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        return self.head(x)

# -------------------------------
# Data Configuration
# -------------------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 2.3)),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Training Functions
# -------------------------------
def train_epoch(model, loader, criterion, optimizer, device, mixup):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, mixed_targets = mixup(inputs, targets)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, mixed_targets)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Calculate accuracy with original labels
        _, predicted = outputs.max(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
        running_loss += loss.item() * inputs.size(0)
        
    return running_loss/total, correct/total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item() * inputs.size(0)
            
    return running_loss/total, correct/total

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == '__main__':
    freeze_support()
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "DATASET"  # Update this path
    num_epochs = 30
    batch_size = 32
    
    # Data Loading
    train_dataset = ImageFolder(os.path.join(dataset_path, "train"), train_transform)
    test_dataset = ImageFolder(os.path.join(dataset_path, "test"), test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model Setup
    model = ResEmoteNetV2(num_classes=7).to(device)
    train_criterion = SoftTargetCrossEntropy().to(device)
    val_criterion = nn.CrossEntropyLoss().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=num_epochs, 
                         steps_per_epoch=len(train_loader))
    mixup = Mixup(alpha=0.4, prob=0.5, num_classes=7)
    
    # Training Loop
    best_acc = 0.0
    for epoch in range(num_epochs):
        start = time.time()
        
        # Train with mixup
        train_loss, train_acc = train_epoch(
            model, train_loader, train_criterion, optimizer, device, mixup
        )
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, val_criterion, device)
        scheduler.step()
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
        
        # Progress reporting
        epoch_time = time.time() - start
        print(f"Epoch {epoch+1}/{num_epochs} [{epoch_time:.1f}s]")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}\n")
    
    print(f"Best Validation Accuracy: {best_acc:.2%}")