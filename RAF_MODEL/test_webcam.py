import torch
import torch.nn as nn
import cv2
import numpy as np
from main import ResEmoteNet  
from torchvision import transforms

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResEmoteNet(num_classes=7).to(device)
model.load_state_dict(torch.load("best_resemotenet_rafdb.pth", map_location=device))
model.eval()

# Define transformation for input images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels (modify based on your dataset)
class_labels = ["Neutral", "Happy", "Sad", "Surprise", "Angry", "Disgust", "Fear"]

# Start webcam feed
cap = cv2.VideoCapture(0)  # 0 is for default webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB and apply transformations
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(device)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        label = class_labels[predicted.item()]

    # Display prediction
    cv2.putText(frame, f"Emotion: {label}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Emotion Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
