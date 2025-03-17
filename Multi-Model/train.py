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
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from src.config import (
    EPOCHS, BATCH_SIZE, MODELS_DIR, RESULTS_DIR,
    FER_MODEL_PATH, VER_MODEL_PATH, FUSION_MODEL_PATH,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_PATIENCE, MIN_LEARNING_RATE, USE_WANDB
)
from src.data_processors import load_fer_data, load_ver_data, check_directory_structure
from src.fer_model import build_fer_model
from src.ver_model import build_ver_model
from src.fusion_model import build_fusion_model
from src.utils import plot_training_history  # Assuming you have a utils module for plotting

# Optional: integrate Weights & Biases logging if enabled
if USE_WANDB:
    import wandb
    from wandb.keras import WandbCallback
    wandb.init(project="multimodal_emotion_recognition", config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE
    })

def calculate_class_weights(y):
    """Calculate class weights for imbalanced datasets"""
    # Assuming y is one-hot encoded; calculate weights based on class frequency
    class_counts = np.sum(y, axis=0)
    total_samples = np.sum(class_counts)
    class_weights = {i: total_samples / (len(class_counts) * class_counts[i])
                     for i in range(len(class_counts))}
    return class_weights

def train_fer_model():
    """Train the Facial Emotion Recognition model with enhanced data augmentation and regularization."""
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
    
    callbacks = [
        checkpoint,
        EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR, 
                          patience=REDUCE_LR_PATIENCE, min_lr=MIN_LEARNING_RATE, verbose=1)
    ]
    if USE_WANDB:
        callbacks.append(WandbCallback())

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    plot_training_history(history, 'fer_training_history.png', 'FER Model')
    return model

def train_ver_model():
    """Train the Voice Emotion Recognition model with class weights and refined architecture."""
    print("Training VER model...")
    train_dataset, val_dataset, _, y_val = load_ver_data()
    
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
    
    callbacks = [
        checkpoint,
        EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR,
                          patience=REDUCE_LR_PATIENCE, min_lr=MIN_LEARNING_RATE, verbose=1)
    ]
    if USE_WANDB:
        from wandb.keras import WandbCallback
        callbacks.append(WandbCallback())
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    plot_training_history(history, 'ver_training_history.png', 'VER Model')
    return model

def train_fusion_model():
    """Train the multi-modal fusion model combining FER and VER embeddings."""
    print("Training Fusion model...")
    # Load the pre-trained FER and VER models
    fer_model = tf.keras.models.load_model(FER_MODEL_PATH)
    ver_model, ver_feature_extractor = build_ver_model()
    ver_model.load_weights(VER_MODEL_PATH)
    
    # Extract embedding layers by name (or via model API)
    fer_embedding_model = tf.keras.Model(
        inputs=fer_model.input,
        outputs=fer_model.get_layer('fer_embedding').output
    )
    ver_embedding_model = tf.keras.Model(
        inputs=ver_model.input,
        outputs=ver_model.get_layer('ver_embedding').output
    )
    
    # Get data generators/datasets
    fer_train_gen, fer_val_gen = load_fer_data()
    ver_train_dataset, ver_val_dataset, _, _ = load_ver_data()
    
    # Prepare fusion data: here we assume that FER generators provide labels
    def prepare_fusion_data(fer_gen, ver_dataset):
        # Generate embeddings for FER images
        fer_embs = fer_embedding_model.predict(fer_gen, verbose=1)
        # Generate embeddings for VER audio from dataset (assuming batch processing)
        ver_embs = []
        for batch in ver_dataset:
            X_batch, _ = batch
            emb_batch = ver_embedding_model.predict(X_batch)
            ver_embs.append(emb_batch)
        ver_embs = np.concatenate(ver_embs, axis=0)
        
        min_samples = min(len(fer_embs), len(ver_embs))
        fer_embs, ver_embs = fer_embs[:min_samples], ver_embs[:min_samples]
        labels = to_categorical(fer_gen.labels[:min_samples], num_classes=fer_embs.shape[-1])
        return [fer_embs, ver_embs], labels
    
    train_inputs, train_labels = prepare_fusion_data(fer_train_gen, ver_train_dataset)
    val_inputs, val_labels = prepare_fusion_data(fer_val_gen, ver_val_dataset)
    
    print(f"Fusion Train Inputs shapes: {np.array(train_inputs).shape}, Labels shape: {train_labels.shape}")
    
    fusion_model = build_fusion_model()
    fusion_model.summary()
    
    checkpoint = ModelCheckpoint(
        FUSION_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    callbacks = [
        checkpoint,
        EarlyStopping(monitor='val_accuracy', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR,
                          patience=REDUCE_LR_PATIENCE, min_lr=MIN_LEARNING_RATE, verbose=1)
    ]
    if USE_WANDB:
        from wandb.keras import WandbCallback
        callbacks.append(WandbCallback())
    
    history = fusion_model.fit(
        train_inputs, train_labels,
        validation_data=(val_inputs, val_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    plot_training_history(history, 'fusion_training_history.png', 'Fusion Model')
    return fusion_model

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    check_directory_structure()
    
    # GPU configuration: Enable memory growth for a smoother training experience.
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("GPU(s) detected and configured.")
        except Exception as e:
            print("GPU configuration failed:", e)
    else:
        print("No GPU available, running on CPU.")
    
    # Uncomment the desired training functions as needed
    # fer_model = train_fer_model()
    # ver_model = train_ver_model()
    fusion_model = train_fusion_model()
    
    print("Training complete. Models saved at:", MODELS_DIR)
