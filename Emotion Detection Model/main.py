import os
import tensorflow as tf
import numpy as np
from scripts.train_model import build_optimized_model, get_callbacks
from scripts.preprocess_data import load_and_preprocess_data
from scripts.evaluate_model import plot_training_history, evaluate_model

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU is available. Using GPU for training.")
    else:
        print("No GPU found. Training will use CPU and RAM only.")

def train_and_evaluate():
    check_gpu()
    
    # Settings
    BATCH_SIZE = 64
    EPOCHS = 100  # You can increase this value for better results, with early stopping it will stop when improvement stalls.
    DATASET_PATH = 'dataset/'
    MODEL_PATH = 'models/optimized_emotion_model.keras'
    
    os.makedirs('models', exist_ok=True)
    
    # Preprocess data: using grayscale images with target size (56, 56)
    train_dataset, val_dataset = load_and_preprocess_data(DATASET_PATH, BATCH_SIZE)
    
    # Build optimized model (input shape now (56,56,1))
    model = build_optimized_model()
    
    # Compile model
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Get callbacks (model checkpoint and early stopping)
    callbacks = get_callbacks(MODEL_PATH)
    
    # Train the model and store the history
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    evaluate_model(model, val_dataset)
    
def main():
    train_and_evaluate()

if __name__ == "__main__":
    main()