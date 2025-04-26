import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR # Or ReduceLROnPlateau
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler # Added SubsetRandomSampler
from torchvision.datasets import ImageFolder
import os
import time
import torch
import numpy as np # Added numpy for potential split
from multiprocessing import freeze_support

# --- Data Augmentation ---
# Stronger Augmentations often help
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(15), # Added Rotation
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), # Slightly more aggressive crop
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # Stronger Jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False) # Added Random Erasing
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)), # Resize slightly larger then center crop
    transforms.CenterCrop(224),   # Standard practice for testing
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Dataset Loading ---
dataset_path = "DATASET" # Ensure this path is correct
train_dataset = ImageFolder(root=os.path.join(dataset_path, "train"), transform=train_transform)
test_dataset = ImageFolder(root=os.path.join(dataset_path, "test"), transform=test_transform)

# Optional: Create a validation split from the training set if you need one
# validation_split = 0.1 # e.g., 10% for validation
# shuffle_dataset = True
# random_seed= 42

# dataset_size = len(train_dataset)
# indices = list(range(dataset_size))
# split = int(np.floor(validation_split * dataset_size))
# if shuffle_dataset :
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]

# # Creating PT data samplers and loaders:
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)

# train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=4, pin_memory=True)
# validation_loader = DataLoader(train_dataset, batch_size=32, sampler=valid_sampler, num_workers=4, pin_memory=True) # Use this for tuning

# If not using a validation split, use the original loaders:
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True) # Increased num_workers if possible
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True) # Increased num_workers if possible

# --- Model Selection ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7 # RAF-DB has 7 basic emotions

# Option A: Pre-trained ResNet-50 (A common strong baseline)
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes) # Replace the final layer

# Option B: Pre-trained EfficientNet (Often more efficient and accurate)
# model = models.efficientnet_b0(pretrained=True) # Or b1, b2, b3... b4 might be a good balance
# num_ftrs = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(num_ftrs, num_classes)

# Option C: Pre-trained Vision Transformer (More recent, can be very powerful)
# model = models.vit_b_16(pretrained=True)
# num_ftrs = model.heads.head.in_features
# model.heads.head = nn.Linear(num_ftrs, num_classes)


model = model.to(device)

# --- Training Setup ---
criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Added Label Smoothing

# Optimizer: AdamW is often preferred for fine-tuning transformers and modern CNNs
# Use a lower learning rate for fine-tuning
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2) # Lower LR, AdamW, higher WD

# Scheduler: CosineAnnealingLR is good, maybe train longer? Or ReduceLROnPlateau
num_epochs = 100 # Increase epochs, fine-tuning might need more time
# scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6) # Add eta_min
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, verbose=True) # Alternative: reduce LR on validation loss plateau


# --- Training and Validation Functions (Keep yours, they are fine) ---
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
        total_correct += torch.sum(preds == labels.data).item() # Use labels.data
        total_samples += labels.size(0) # Use labels.size(0)


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
            total_correct += torch.sum(preds == labels.data).item() # Use labels.data
            total_samples += labels.size(0) # Use labels.size(0)
    val_loss = running_loss / total_samples
    val_acc = total_correct / total_samples
    return val_loss, val_acc

# --- Main Training Loop ---
if __name__ == '__main__':
    # freeze_support() # Usually needed for multiprocessing on Windows

    print("Starting fine-tuning...")
    print(f"Using device: {device}")
    print("Train dataset size:", len(train_dataset))
    # if 'validation_loader' in locals(): print("Validation dataset size:", len(val_indices)) # If using split
    print("Test dataset size:", len(test_dataset))

    best_val_acc = 0.0
    patience_counter = 0 # For ReduceLROnPlateau scheduler or early stopping
    patience_limit = 15 # Stop if val acc doesn't improve for 15 epochs after LR reduction

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # If using a validation split:
        # val_loss, val_acc = validate(model, validation_loader, criterion, device)
        # If using the test set for validation (less ideal, but common if no separate val set):
        val_loss, val_acc = validate(model, test_loader, criterion, device) # Using test set as validation here

        # Adjust LR based on validation loss for ReduceLROnPlateau
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
             scheduler.step(val_loss)
        # Adjust LR every epoch for CosineAnnealingLR
        elif isinstance(scheduler, CosineAnnealingLR):
             scheduler.step()


        end_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {end_time - start_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f} (Best: {best_val_acc:.4f})")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Current LR: {current_lr:.6f}")


        if val_acc > best_val_acc:
            print(f"  Validation accuracy improved ({best_val_acc:.4f} --> {val_acc:.4f}). Saving model...")
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_pretrained_rafdb.pth")
            patience_counter = 0 # Reset patience
        else:
            patience_counter += 1
            print(f"  Validation accuracy did not improve. Patience: {patience_counter}/{patience_limit}")


        # Optional: Early Stopping
        # if patience_counter >= patience_limit:
        #     print("Early stopping triggered.")
        #     break

    print(f"Training finished. Best Validation Accuracy: {best_val_acc:.4f}")

    # Optional: Load best model and evaluate on test set if you used a separate validation set during training
    # if 'validation_loader' in locals():
    #     print("Loading best model for final test evaluation...")
    #     model.load_state_dict(torch.load("best_pretrained_rafdb.pth"))
    #     test_loss, test_acc = validate(model, test_loader, criterion, device)
    #     print(f"Final Test Set Performance: Loss={test_loss:.4f}, Accuracy={test_acc:.4f}")