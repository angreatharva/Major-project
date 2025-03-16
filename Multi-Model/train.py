import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from src.config import (
    EPOCHS, BATCH_SIZE, MODELS_DIR, RESULTS_DIR,
    FER_MODEL_PATH, VER_MODEL_PATH, FUSION_MODEL_PATH,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_PATIENCE, MIN_LEARNING_RATE, NUM_CLASSES
)
from src.data_processors import load_fer_data, load_ver_data, check_directory_structure
from src.fer_model import build_fer_model
from src.ver_model import build_ver_model
from src.fusion_model import build_fusion_model
from src.utils import plot_training_history

def calculate_class_weights(y):
    """Calculate class weights for imbalanced datasets"""
    class_counts = np.sum(y, axis=0)
    total_samples = np.sum(class_counts)
    # (Implementation can be added if needed)
    
def train_fer_model():
    """Train the Facial Emotion Recognition model"""
    print("Training FER model...")
    train_generator, val_generator = load_fer_data()
    model = build_fer_model()
    model.summary()
    
    checkpoint = ModelCheckpoint(
        FER_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[
            checkpoint,
            EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE),
            ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR, 
                            patience=REDUCE_LR_PATIENCE, min_lr=MIN_LEARNING_RATE)
        ]
    )
    
    plot_training_history(history, 'fer_training_history.png', 'FER Model')
    return model

def train_ver_model():
    """Enhanced VER training with class weights"""
    print("Training VER model...")
    train_generator, val_generator, _, y_val = load_ver_data()
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_val)
    
    model, _ = build_ver_model()
    model.summary()
    
    checkpoint = ModelCheckpoint(
        VER_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[
            checkpoint,
            EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE),
            ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR,
                            patience=REDUCE_LR_PATIENCE, min_lr=MIN_LEARNING_RATE)
        ]
    )
    
    plot_training_history(history, 'ver_training_history.png', 'VER Model')
    return model

def train_fusion_model():
    """Train the multi-modal fusion model"""
    print("Training fusion model...")
    fer_model = tf.keras.models.load_model(FER_MODEL_PATH)
    ver_model = tf.keras.models.load_model(VER_MODEL_PATH)
    
    fer_embedding_model = tf.keras.Model(
        inputs=fer_model.input,
        outputs=fer_model.get_layer('fer_embedding').output
    )
    ver_embedding_model = tf.keras.Model(
        inputs=ver_model.input,
        outputs=ver_model.get_layer('ver_embedding').output
    )
    
    fer_train_gen, fer_val_gen = load_fer_data()
    ver_train_gen, ver_val_gen, _, _ = load_ver_data()
    
    fusion_model = build_fusion_model()
    fusion_model.summary()
    
    def prepare_fusion_data(fer_gen, ver_gen):
        fer_embs = fer_embedding_model.predict(fer_gen)
        ver_embs = ver_embedding_model.predict(ver_gen)
        
        # Ensure both embeddings have the same number of samples
        min_samples = min(len(fer_embs), len(ver_embs))
        fer_embs, ver_embs = fer_embs[:min_samples], ver_embs[:min_samples]
        
        # One-hot encode labels
        labels = to_categorical(fer_gen.labels[:min_samples], NUM_CLASSES)
        
        return [fer_embs, ver_embs], labels
    
    train_inputs, train_labels = prepare_fusion_data(fer_train_gen, ver_train_gen)
    val_inputs, val_labels = prepare_fusion_data(fer_val_gen, ver_val_gen)
    
    print(f"Train Inputs: {np.array(train_inputs).shape}, Labels: {train_labels.shape}")
    
    checkpoint = ModelCheckpoint(
        FUSION_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    history = fusion_model.fit(
        train_inputs, train_labels,
        validation_data=(val_inputs, val_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            checkpoint,
            EarlyStopping(monitor='val_accuracy', patience=EARLY_STOPPING_PATIENCE),
            ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR,
                            patience=REDUCE_LR_PATIENCE, min_lr=MIN_LEARNING_RATE)
        ]
    )
    
    plot_training_history(history, 'fusion_training_history.png', 'Fusion Model')
    return fusion_model

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    check_directory_structure()
    
    # GPU configuration
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("GPU detected and configured")
        except:
            print("Could not configure GPU")
    else:
        print("No GPU available")
    
    # Train models (uncomment the desired training functions)
    # fer_model = train_fer_model()
    # ver_model = train_ver_model()
    fusion_model = train_fusion_model()
    
    print("Training complete. Models saved at:", MODELS_DIR)
