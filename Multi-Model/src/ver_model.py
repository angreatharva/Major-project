import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU,
    MaxPool2D, Dropout, Reshape, GRU, Dense,
    Bidirectional, LayerNormalization, Attention
)
from src.config import NUM_CLASSES, EMBEDDING_DIM, AUDIO_N_MELS

def build_ver_model():
    """CRNN model with enhanced regularization"""
    input_shape = (None, AUDIO_N_MELS, 1)
    inputs = Input(shape=input_shape, name='audio_input')

    # CNN Block with L2 regularization
    x = Conv2D(64, (3, 3), padding='same', 
              kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, (3, 3), padding='same',
              kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, (3, 3), padding='same',
              kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.4)(x)

    # Reshape for RNN
    x = Reshape((-1, x.shape[2] * x.shape[3]))(x)

    # BiGRU with Attention
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = LayerNormalization()(x)
    
    # Attention layer
    context = Attention()([x, x])
    x = tf.concat([x, context], axis=-1)
    
    x = Bidirectional(GRU(128))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.5)(x)

    # Output layers
    embeddings = Dense(EMBEDDING_DIM, activation='relu', 
                      kernel_regularizer=tf.keras.regularizers.l2(0.001),
                      name='ver_embedding')(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(embeddings)

    model = Model(inputs, outputs)
    feature_extractor = Model(inputs, embeddings)
    
    # Compile with adjusted learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, feature_extractor
