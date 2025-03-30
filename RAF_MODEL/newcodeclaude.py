import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.datasets import ImageFolder
from multiprocessing import freeze_support
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import random

# For reproducibility
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# -------------------------------
# Squeeze-Excitation (SE) Block with higher reduction ratio
# -------------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):  # Increased attention capacity with lower reduction
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Added max pooling for better feature selection
        self.fc = nn.Sequential(
            nn.Conv2d(channel*2, channel // reduction, kernel_size=1),  # Using Conv2d instead of Linear
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.fc(y)
        return x * y

# -------------------------------
# Improved Residual Block with SE and Dropout
# -------------------------------
class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=8, dropout_rate=0.2):
        super(ResidualSEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction)
        self.dropout = nn.Dropout2d(dropout_rate)  # Adding spatial dropout
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)  # Apply dropout after first activation
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out

# -------------------------------
# Enhanced ResEmoteNet Architecture
# -------------------------------
class ResEmoteNet(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.3):
        super(ResEmoteNet, self).__init__()
        # Initial convolution: input size (B,3,224,224)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Four stages of residual blocks with more blocks in deeper layers
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2, dropout_rate=dropout_rate)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Adding attention before final classification
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )
        
        # Two FC layers with dropout for better generalization
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout_rate):
        layers = []
        layers.append(ResidualSEBlock(in_channels, out_channels, stride, dropout_rate=dropout_rate))
        for _ in range(1, blocks):
            layers.append(ResidualSEBlock(out_channels, out_channels, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # (B,64,H/2,W/2)
        x = self.maxpool(x)                     # (B,64,H/4,W/4)
        x = self.layer1(x)                      # (B,64, ...)
        x = self.layer2(x)                      # (B,128, ...)
        x = self.layer3(x)                      # (B,256, ...)
        x = self.layer4(x)                      # (B,512, ...)
        x = self.avgpool(x)                     # (B,512,1,1)
        x = torch.flatten(x, 1)                 # (B,512)
        
        # Apply attention mechanism
        att = self.attention(x)
        x = x * att
        
        # Two FC layers with dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------------------
# Enhanced Data Augmentation and DataLoaders
# -------------------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3)
])

test_transform = transforms.Compose([
    transforms.Resize((240, 240)),  # Slightly larger size
    transforms.CenterCrop(224),     # Then center crop for consistent testing
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Adjust the 'dataset_path' below to your local RAF‑DB dataset location
dataset_path = "DATASET"  # Folder with 'train' and 'test' subdirectories

# -------------------------------
# Learning Rate Finder
# -------------------------------
def find_lr(model, train_loader, criterion, optimizer, device, start_lr=1e-7, end_lr=10, num_iter=100):
    """Find the optimal learning rate for the model."""
    model.train()
    lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)
    losses = []
    
    # Save initial parameters
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Set learning rate to start value
    for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
    
    for i, (images, labels) in enumerate(train_loader):
        if i >= num_iter:
            break
        
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lrs[i]
    
    # Restore initial parameters
    model.load_state_dict(initial_state)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.savefig('lr_finder.png')
    
    # Find best learning rate (highest drop in loss)
    smoothed_losses = np.array(losses)
    gradients = np.gradient(smoothed_losses)
    steepest_idx = np.argmin(gradients)
    best_lr = lrs[steepest_idx] / 10  # Divide by 10 for safety
    print(f"Recommended learning rate: {best_lr:.7f}")
    return best_lr

# -------------------------------
# Class-balanced Sampling
# -------------------------------
def create_weighted_sampler(dataset):
    """Create a weighted sampler to handle class imbalance."""
    targets = [label for _, label in dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    weights = class_weights[targets]
    return WeightedRandomSampler(weights, len(weights))

# -------------------------------
# Updated Training and Validation Functions with Mixup Augmentation
# -------------------------------
def mixup_data(x, y, alpha=0.2, device='cpu'):
    """Performs mixup: creates convex combinations of pairs of examples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Computes mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_epoch(model, loader, criterion, optimizer, device, use_mixup=True, mixup_alpha=0.4):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        # Apply mixup augmentation
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha, device)
            
        optimizer.zero_grad()
        outputs = model(images)
        
        if use_mixup:
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            # For accuracy calculation with mixup, we'll use the first labels only
            _, preds = torch.max(outputs, 1)
            total_correct += torch.sum(preds == labels_a).item() * lam + torch.sum(preds == labels_b).item() * (1 - lam)
        else:
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            total_correct += torch.sum(preds == labels).item()
            
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
            
    epoch_loss = running_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += images.size(0)
            
            # Store predictions and labels for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = running_loss / total_samples
    val_acc = total_correct / total_samples
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, output_dict=True)
    
    return val_loss, val_acc, conf_matrix, class_report

# -------------------------------
# Test Time Augmentation (TTA)
# -------------------------------
def test_time_augmentation(model, image, device, num_augs=5):
    """Apply test-time augmentation to improve prediction accuracy."""
    model.eval()
    # Define TTA transforms
    tta_transforms = [
        transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                std=[1/0.229, 1/0.224, 1/0.225]),  # Unnormalize
            transforms.ToPILImage(),
            transforms.Resize((240, 240)),
            transforms.CenterCrop(224) if i == 0 else transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5 if i != 0 else 0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) for i in range(num_augs)
    ]
    
    # Apply transforms and average predictions
    with torch.no_grad():
        outputs = []
        # Original image is already normalized, unnormalize first
        img = image.clone()
        for transform in tta_transforms:
            aug_img = transform(img.cpu()[0]).unsqueeze(0).to(device)
            output = model(aug_img)
            outputs.append(F.softmax(output, dim=1))
        
        # Average predictions
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        _, pred = torch.max(avg_output, 1)
        
    return pred.item()

# -------------------------------
# Main Training Loop with Early Stopping and Learning Rate Finder
# -------------------------------
if __name__ == '__main__':
    freeze_support()
    
    # Load datasets
    train_dataset = ImageFolder(root=os.path.join(dataset_path, "train"), transform=train_transform)
    test_dataset = ImageFolder(root=os.path.join(dataset_path, "test"), transform=test_transform)
    
    # Create weighted sampler for handling class imbalance
    train_sampler = create_weighted_sampler(train_dataset)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = ResEmoteNet(num_classes=7, dropout_rate=0.3).to(device)
    
    # Cross-entropy loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Initialize optimizer (will update learning rate after finding optimal lr)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4, betas=(0.9, 0.999))
    
    print("Starting training...")
    print(f"Device: {device}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Class distribution: {np.bincount([label for _, label in train_dataset.samples])}")
    
    # Find optimal learning rate
    print("Finding optimal learning rate...")
    optimal_lr = find_lr(model, train_loader, criterion, optimizer, device)
    
    # Reset optimizer with found learning rate
    optimizer = optim.AdamW(model.parameters(), lr=optimal_lr, weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Use OneCycleLR scheduler instead of CosineAnnealingLR
    num_epochs = 50  # Increase epochs to give model more time to learn
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=optimal_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.1,  # Warm-up for 10% of training
        div_factor=25,  # Initial LR = max_lr/25
        final_div_factor=1000  # Final LR = max_lr/1000/25
    )
    
    # Early stopping variables
    best_val_acc = 0.0
    patience = 7
    patience_counter = 0
    
    # Training history for plotting
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train with mixup (gradually decrease alpha throughout training)
        mixup_alpha = max(0.4 * (1 - epoch/num_epochs), 0.1)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, 
                                           use_mixup=True, mixup_alpha=mixup_alpha)
        
        # Validate with detailed metrics
        val_loss, val_acc, conf_matrix, class_report = validate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Record history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print epoch results
        end_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} completed in {end_time - start_time:.2f} seconds")
        print(f"LR: {current_lr:.7f} | Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Print per-class metrics
        print("\nPer-class performance:")
        for cls in range(7):
            if str(cls) in class_report:
                print(f"Class {cls}: Precision={class_report[str(cls)]['precision']:.4f}, "
                      f"Recall={class_report[str(cls)]['recall']:.4f}, "
                      f"F1-score={class_report[str(cls)]['f1-score']:.4f}")
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}. Saving model...")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, "best_resemotenet_rafdb_claude.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    # Load best model and evaluate
    checkpoint = torch.load("best_resemotenet_rafdb_claude.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation with TTA
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    print("\nPerforming final evaluation with Test-Time Augmentation...")
    with torch.no_grad():
        for images, labels in test_loader:
            batch_size = images.size(0)
            correct_batch = 0
            
            # Process each image with TTA
            for i in range(batch_size):
                img = images[i:i+1].to(device)
                label = labels[i].item()
                
                # Apply TTA
                pred = test_time_augmentation(model, img, device, num_augs=10)
                if pred == label:
                    correct_batch += 1
                
                all_preds.append(pred)
                all_labels.append(label)
            
            total_correct += correct_batch
            total_samples += batch_size
    
    final_acc = total_correct / total_samples
    print(f"Final accuracy with TTA: {final_acc:.4f}")
    
    # Compute final confusion matrix and classification report
    final_conf_matrix = confusion_matrix(all_labels, all_preds)
    final_class_report = classification_report(all_labels, all_preds)
    
    print("\nFinal Classification Report:")
    print(final_class_report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(final_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    print("Training completed!")