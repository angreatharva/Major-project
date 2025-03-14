from tensorflow.keras.models import load_model
import numpy as np

# Load trained model
model = load_model("../models/multimodal_model.h5")

# Load new face & audio data (processed)
X_face = np.load("new_face.npy")
X_audio = np.load("new_audio.npy")

# Predict Emotion
prediction = model.predict([X_face, X_audio])
print("Predicted Emotion:", np.argmax(prediction))
