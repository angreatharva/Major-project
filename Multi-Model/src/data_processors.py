import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from src.config import (
    BATCH_SIZE, IMAGE_SIZE, NUM_CLASSES,
    AUDIO_SAMPLE_RATE, AUDIO_DURATION, AUDIO_N_MELS,
    AUDIO_N_FFT, AUDIO_HOP_LENGTH
)

def check_directory_structure():
    """Ensure required directories exist."""
    required_dirs = [
        'datasets/FER-Dataset/DATASET/train',
        'datasets/FER-Dataset/DATASET/test',
        'datasets/VER-Dataset/audio_speech_actors_01-24',
        'models',
        'results'
    ]
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)

def map_ravdess_label(filename):
    """
    Extract emotion label from RAVDESS filename.
    Format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
    Emotion mapping: 01=Neutral, 03=Happy, 04=Sad, 05=Angry, 06=Fear, 07=Disgust, 08=Surprise
    """
    emotion_str = filename.split('-')[2]
    emotion_map = {
        '01': 0, '03': 4, '04': 5, '05': 1,
        '06': 3, '07': 2, '08': 6
    }
    return emotion_map.get(emotion_str, 0)

def load_fer_data():
    """Load FER dataset with advanced augmentation and preprocessing."""
    train_dir = 'datasets/FER-Dataset/DATASET/train'
    test_dir = 'datasets/FER-Dataset/DATASET/test'

    def preprocess_image(x):
        # Convert to grayscale if needed
        if len(x.shape) == 2 or x.shape[-1] == 1:
            return x[..., np.newaxis] if len(x.shape) == 2 else x
        elif x.shape[-1] == 3:
            import cv2
            return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        else:
            raise ValueError(f"Unexpected image shape: {x.shape}")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=preprocess_image
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=preprocess_image
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale'
    )

    val_generator = val_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale'
    )

    return train_generator, val_generator

def extract_audio_features(file_path):
    """Extract and preprocess Mel spectrogram features from audio files."""
    try:
        import librosa
        y, sr = librosa.load(file_path, sr=AUDIO_SAMPLE_RATE)
        y, _ = librosa.effects.trim(y, top_db=30)
        
        # Audio augmentation: noise injection and time stretching
        if np.random.rand() > 0.5:
            y += np.random.normal(0, 0.002, y.shape)
        if np.random.rand() > 0.5:
            y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.9, 1.1))
        
        y = np.pad(y, (0, max(0, AUDIO_SAMPLE_RATE * AUDIO_DURATION - len(y))), 'constant')
        
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=AUDIO_N_MELS,
            n_fft=AUDIO_N_FFT, hop_length=AUDIO_HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)
        return mel_norm.T[..., np.newaxis]
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_ver_data(test_size=0.2, random_state=42):
    """Load and process VER dataset into TensorFlow datasets."""
    ver_path = 'datasets/VER-Dataset/audio_speech_actors_01-24'
    if not os.path.exists(ver_path):
        raise FileNotFoundError("VER dataset not found in the expected location.")
    
    audio_files, labels = [], []
    for root, _, files in os.walk(ver_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                label = map_ravdess_label(file)
                audio_files.append(file_path)
                labels.append(label)

    y = to_categorical(labels, NUM_CLASSES)
    files_train, files_test, y_train, y_test = train_test_split(
        audio_files, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print("Processing training audio...")
    X_train = [extract_audio_features(f) for f in files_train]
    X_train = [x for x in X_train if x is not None]
    print("Processing test audio...")
    X_test = [extract_audio_features(f) for f in files_test]
    X_test = [x for x in X_test if x is not None]

    max_time = max(x.shape[0] for x in X_train + X_test)

    def pad_features(features):
        padded = np.zeros((max_time, AUDIO_N_MELS, 1))
        padded[:features.shape[0]] = features[:max_time]
        return padded

    X_train = np.array([pad_features(x) for x in X_train])
    X_test = np.array([pad_features(x) for x in X_test])

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset, X_test, y_test
