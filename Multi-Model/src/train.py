import numpy as np
from tensorflow.keras.optimizers import Adam
from multimodal_model import multimodal_model

# Load preprocessed data
X_faces = np.load("../datasets/X_faces.npy")
y_faces = np.load("../datasets/y_faces.npy")
X_audio = np.load("../datasets/X_audio.npy")
y_audio = np.load("../datasets/y_audio.npy")

# Train model
multimodal_model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
multimodal_model.fit([X_faces, X_audio], y_faces, epochs=10, batch_size=16, validation_split=0.2)

# Save model
multimodal_model.save("../models/multimodal_model.h5")
