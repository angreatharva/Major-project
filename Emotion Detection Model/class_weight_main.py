import os
import numpy as np
import tensorflow as tf
from scripts.preprocess_data import load_and_preprocess_data
from scripts.train_model import create_emotion_model, get_callbacks, compile_model
from scripts.evaluate_model import plot_training_history, evaluate_model
from sklearn.utils.class_weight import compute_class_weight

def main():
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 50
    MODEL_PATH = 'models/emotion_weighted_model.keras'
    DATASET_PATH = 'dataset/'
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_dataset, val_dataset = load_and_preprocess_data(DATASET_PATH, BATCH_SIZE)
    
    # Calculate class weights to handle imbalanced data
    print("Calculating class weights...")
    classes = np.unique(train_dataset.classes)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_dataset.classes
    )
    class_weight_dict = dict(zip(classes, class_weights))
    print("Class weights:", class_weight_dict)
    
    # Create and compile model
    print("Creating model...")
    model = create_emotion_model()
    compile_model(model)
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Train the model
    print("\nTraining model with class weights...")
    callbacks = get_callbacks(MODEL_PATH)
    
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        class_weight=class_weight_dict,  # Add class weights
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
    evaluate_model(model, val_dataset)
    
    print("\nTraining and evaluation completed!")
    print(f"Best model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    main()