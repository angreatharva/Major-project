import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_and_preprocess_data(dataset_path, batch_size):
    IMG_SIZE = (48, 48)
    data_gen = ImageDataGenerator(
        rescale=1./255
    )
    
    # Load training data only
    train_dataset = data_gen.flow_from_directory(
        dataset_path + 'training_dataset/',  # Path to training data
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical'  # Multi-class classification
    )
    
    return train_dataset
