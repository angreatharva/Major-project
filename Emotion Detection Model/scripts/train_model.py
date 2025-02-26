import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, 
    Conv2D, MaxPooling2D, Flatten, Add, Input
)
from tensorflow.keras.models import Model

def create_emotion_model():
    inputs = Input(shape=(48, 48, 3))
    
    # First Convolutional Block
    x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    shortcut1 = x  # Store for residual connection
    x = MaxPooling2D(pool_size=(2,2))(x)  # Now shape reduced (48,48)->(24,24)
    
    # Second Convolutional Block
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    # Adjust shortcut1 to match the current shape
    shortcut1 = Conv2D(128, (1,1), activation='relu', padding='same')(shortcut1)
    shortcut1 = MaxPooling2D(pool_size=(2,2))(shortcut1)  # Now shortcut1 shape should match x
    x = Add()([x, shortcut1])  # Residual connection
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    # Third Convolutional Block
    shortcut2 = x  # Store the input for the block
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    # Adjust shortcut2 to match the shape of x before addition
    shortcut2 = Conv2D(256, (1,1), activation='relu', padding='same')(shortcut2)
    x = Add()([x, shortcut2])  # Residual connection
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    # Fully Connected Layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(7, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

def create_transfer_learning_model():
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(7, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

def get_callbacks(model_path):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6  # Decrease LR on plateau
        )
    ]
