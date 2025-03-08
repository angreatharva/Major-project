import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(dataset_path, batch_size):
    IMG_SIZE = (56, 56)
    
    # Use a validation split of 20%
    train_data_gen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=30,  # Increased rotation range
        width_shift_range=0.2,  # Increased width shift
        height_shift_range=0.2,  # Increased height shift
        brightness_range=[0.2, 1.8],  # Increased brightness variations
        shear_range=0.3,
        zoom_range=0.4,  # Increased zoom range
        horizontal_flip=True,
        validation_split=0.2
    )

    val_data_gen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.2
    )
    
    # Note: setting color_mode to "grayscale"
    train_dataset = train_data_gen.flow_from_directory(
        directory=dataset_path + 'training_dataset/',
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    val_dataset = val_data_gen.flow_from_directory(
        directory=dataset_path + 'training_dataset/',
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_dataset, val_dataset
