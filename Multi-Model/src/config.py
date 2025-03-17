import os
from tensorflow.keras import callbacks

# General configuration
EPOCHS = 100
BATCH_SIZE = 32
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
USE_WANDB = False  # Set to True for Weights & Biases integration
GRADIENT_CLIP = 1.0  # For gradient clipping

# Model paths
FER_MODEL_PATH = os.path.join(MODELS_DIR, 'fer_model.h5')
VER_MODEL_PATH = os.path.join(MODELS_DIR, 'ver_model.h5')
FUSION_MODEL_PATH = os.path.join(MODELS_DIR, 'fusion_model.h5')

# Emotion labels and classes
EMOTION_LABELS = ['Neutral', 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']
NUM_CLASSES = len(EMOTION_LABELS)

# Image parameters for FER
IMAGE_SIZE = (48, 48)
FER_INPUT_SHAPE = (*IMAGE_SIZE, 1)  # Grayscale

# Audio parameters for VER (Mel Spectrogram configuration)
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 3  # seconds
AUDIO_N_MELS = 64    
AUDIO_N_FFT = 1024
AUDIO_HOP_LENGTH = 512
VER_INPUT_SHAPE = (None, AUDIO_N_MELS, 1)

# Embedding dimensions (should be consistent across models)
EMBEDDING_DIM = 128

# Learning rate schedule
INITIAL_LEARNING_RATE = 0.001
REDUCE_LR_FACTOR = 0.2
REDUCE_LR_PATIENCE = 5
MIN_LEARNING_RATE = 1e-6

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 20

# Common callbacks configuration (for reuse in training scripts)
COMMON_CALLBACKS = [
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=VER_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=FER_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]
