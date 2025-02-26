### train_model.py
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Add, Input
from tensorflow.keras.models import Model

def create_emotion_model():
    inputs = Input(shape=(48, 48, 3))
    x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    shortcut = x
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
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