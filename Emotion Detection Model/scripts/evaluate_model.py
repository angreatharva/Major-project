import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf

def plot_training_history(history):
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation accuracy values
    ax1.plot(history.history['accuracy'])
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'])
        ax1.legend(['Train', 'Validation'], loc='upper left')
    else:
        ax1.legend(['Train'], loc='upper left')
    ax1.set_title('Model accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    
    # Plot training & validation loss values
    ax2.plot(history.history['loss'])
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'])
        ax2.legend(['Train', 'Validation'], loc='upper left')
    else:
        ax2.legend(['Train'], loc='upper left')
    ax2.set_title('Model loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model(model, test_dataset):
    # Emotions mapping
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Evaluate model on test dataset
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Get predictions for confusion matrix and classification report
    # Reset the test_dataset to ensure we process all samples
    test_dataset.reset()
    
    # Collect all true labels and predictions
    y_true = []
    y_pred = []
    batch_count = 0
    max_batches = len(test_dataset)
    
    for images, labels in test_dataset:
        batch_count += 1
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(np.argmax(labels, axis=1))
        
        if batch_count >= max_batches:
            break
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=emotions, yticklabels=emotions)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=emotions))
    
    # Plot some example predictions
    test_dataset.reset()
    images, labels = next(test_dataset)
    predictions = model.predict(images, verbose=0)
    
    plt.figure(figsize=(15, 10))
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        true_label = emotions[np.argmax(labels[i])]
        pred_label = emotions[np.argmax(predictions[i])]
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('example_predictions.png')
    plt.show()