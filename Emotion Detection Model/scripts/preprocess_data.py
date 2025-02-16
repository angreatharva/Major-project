import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_and_preprocess_data(dataset_path, batch_size):
    IMG_SIZE = (48, 48)
    data_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  # 20% of the data for validation
    )
    
    # Load training data
    train_dataset = data_gen.flow_from_directory(
        dataset_path + 'training_dataset/',  # Path to training data
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',  # Multi-class classification
        subset='training'
    )
    
    # Load validation data
    test_dataset = data_gen.flow_from_directory(
        dataset_path + 'validation_dataset/',  # Path to validation data
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',  # Multi-class classification
        subset='validation'
    )
    
    return train_dataset, test_dataset
