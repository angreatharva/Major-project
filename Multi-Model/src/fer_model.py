from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU,
    MaxPool2D, Dropout, Add, GlobalAvgPool2D,
    Dense, Multiply, Reshape
)
import tensorflow as tf
from src.config import IMAGE_SIZE, EMBEDDING_DIM, NUM_CLASSES

def channel_attention(input_tensor, ratio=8):
    channel = input_tensor.shape[-1]
    shared_layer = Dense(channel // ratio, activation='relu')
    
    avg_pool = GlobalAvgPool2D()(input_tensor)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer(avg_pool)
    
    max_pool = tf.reduce_max(input_tensor, axis=[1, 2], keepdims=True)
    max_pool = shared_layer(max_pool)
    
    attention = Add()([avg_pool, max_pool])
    attention = Dense(channel, activation='sigmoid')(attention)
    return Multiply()([input_tensor, attention])

def residual_block(x, filters, kernel=3, stride=1):
    shortcut = x
    x = Conv2D(filters, kernel, padding='same', strides=stride)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(filters, kernel, padding='same')(x)
    x = BatchNormalization()(x)
    
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

def build_fer_model():
    """Enhanced FER model with multiple residual blocks and integrated channel attention."""
    inputs = Input(shape=(*IMAGE_SIZE, 1))
    
    x = Conv2D(64, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # First residual block group
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = MaxPool2D(2)(x)
    x = channel_attention(x)
    
    # Second residual block group
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = channel_attention(x)
    
    # Third residual block group
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    x = channel_attention(x)
    
    # Head: global pooling and dense layers with dropout for regularization
    x = GlobalAvgPool2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    embeddings = Dense(EMBEDDING_DIM, activation='relu', name='fer_embedding')(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(embeddings)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
