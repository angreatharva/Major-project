import librosa
import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

# Paths
AUDIO_PATH = "../datasets/voice_clips/"
LABELS_CSV = "../datasets/voice_train.csv"

# Load dataset labels
df = pd.read_csv(LABELS_CSV)
labels = df['emotion'].values
audio_paths = df['audio_path'].values

# Convert audio to Mel-Spectrogram
def preprocess_audio(file_path, target_shape=(128, 128)):
    y, sr = librosa.load(os.path.join(AUDIO_PATH, file_path), sr=16000)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_shape[0])
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = mel_spec / np.max(mel_spec)  # Normalize
    return np.expand_dims(mel_spec, axis=-1)  # (128, 128, 1)

# Process audio clips
X_audio = np.array([preprocess_audio(path) for path in audio_paths])
y_audio = to_categorical(labels, num_classes=7)

# Save preprocessed data
np.save("../datasets/X_audio.npy", X_audio)
np.save("../datasets/y_audio.npy", y_audio)
