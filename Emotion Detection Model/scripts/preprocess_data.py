### preprocess_data.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_and_preprocess_data(dataset_path, batch_size):
    IMG_SIZE = (48, 48)
    train_data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        brightness_range=[0.5, 1.5],
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    val_data_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    train_dataset = train_data_gen.flow_from_directory(
        dataset_path + 'training_dataset/',
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    val_dataset = val_data_gen.flow_from_directory(
        dataset_path + 'training_dataset/',
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return train_dataset, val_dataset