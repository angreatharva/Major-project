import cv2
import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

# Paths
DATASET_PATH = "../datasets/face_images/"
LABELS_CSV = "../datasets/face_train.csv"

# Load dataset labels
df = pd.read_csv(LABELS_CSV)
labels = df['emotion'].values
image_paths = df['image_path'].values

# Image processing
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(os.path.join(DATASET_PATH, image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return img

# Process images
X_faces = np.array([preprocess_image(path) for path in image_paths])
y_faces = to_categorical(labels, num_classes=7)

# Save preprocessed data
np.save("../datasets/X_faces.npy", X_faces)
np.save("../datasets/y_faces.npy", y_faces)
