import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def create_ensemble_model():
    # Load both trained models
    cnn_model = tf.keras.models.load_model('models/emotion_model.keras')
    transfer_model = tf.keras.models.load_model('models/emotion_transfer_model.keras')
    
    return [cnn_model, transfer_model]

def evaluate_ensemble(models, test_dataset):
    # Emotions mapping
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Reset the test_dataset to ensure we process all samples
    test_dataset.reset()
    
    # Collect all ground truth labels
    y_true = []
    all_image_batches = []
    batch_count = 0
    max_batches = len(test_dataset)
    
    for images, labels in test_dataset:
        batch_count += 1
        all_image_batches.append(images)
        y_true.extend(np.argmax(labels, axis=1))
        
        if batch_count >= max_batches:
            break
    
    # Get predictions from each model and combine them
    all_predictions = []
    
    for model in models:
        test_dataset.reset()
        model_predictions = []
        batch_count = 0
        
        for images in all_image_batches:
            batch_preds = model.predict(images, verbose=0)
            model_predictions.extend(batch_preds)
            batch_count += 1
            
            if batch_count >= max_batches:
                break
        
        all_predictions.append(model_predictions)
    
    # Average the predictions from all models
    ensemble_predictions = np.mean(all_predictions, axis=0)
    y_pred = np.argmax(ensemble_predictions, axis=1)
    
    # Calculate ensemble accuracy
    ensemble_accuracy = np.mean(y_pred == y_true)
    print(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=emotions, yticklabels=emotions)
    plt.title('Ensemble Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('ensemble_confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print("\nEnsemble Classification Report:")
    print(classification_report(y_true, y_pred, target_names=emotions))
    
    # Save ensemble predictions for later use
    np.save('ensemble_predictions.npy', ensemble_predictions)
    np.save('true_labels.npy', y_true)
    
    return ensemble_accuracy, y_pred, y_true

if __name__ == "__main__":
    from scripts.preprocess_data import load_and_preprocess_data
    
    # Load the validation dataset
    _, val_dataset = load_and_preprocess_data('dataset/', 32)
    
    # Create ensemble from trained models
    models = create_ensemble_model()
    
    # Evaluate the ensemble
    accuracy, predictions, true_labels = evaluate_ensemble(models, val_dataset)