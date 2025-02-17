import os
import numpy as np
import tensorflow as tf
from scripts.preprocess_data import load_and_preprocess_data
from scripts.train_model import create_emotion_model, get_callbacks, compile_model
from scripts.evaluate_model import plot_training_history, evaluate_model  # Updated Import

def main():
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 50
    MODEL_PATH = 'models/emotion_model.keras'  # Changed to .keras format
    DATASET_PATH = 'dataset/'
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_dataset = load_and_preprocess_data(DATASET_PATH, BATCH_SIZE)
    
    # Create and compile model
    print("Creating model...")
    model = create_emotion_model()
    compile_model(model)
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Train the model
    print("\nTraining model...")
    callbacks = get_callbacks(MODEL_PATH)
    
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    
    # Load the best model (the one with best validation accuracy)
    print("\nLoading best model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print("\nPerforming model evaluation...")
    evaluate_model(model, test_dataset)  # Updated Function Call
    
    print("\nTraining and evaluation completed!")
    print(f"Best model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    main()
