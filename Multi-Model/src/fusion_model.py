import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Concatenate, Input, Activation
from src.config import NUM_CLASSES, EMBEDDING_DIM

def build_fusion_model():
    """
    Build the multi-modal fusion model that combines FER and VER embeddings
    
    Returns:
        model: Compiled Keras model
    """
    # Input layers for embeddings
    fer_input = Input(shape=(EMBEDDING_DIM,), name='fer_embedding_input')
    ver_input = Input(shape=(EMBEDDING_DIM,), name='ver_embedding_input')
    
    # Process each input stream separately first
    fer_processed = Dense(64, activation='relu')(fer_input)
    fer_processed = BatchNormalization()(fer_processed)
    
    ver_processed = Dense(64, activation='relu')(ver_input)
    ver_processed = BatchNormalization()(ver_processed)
    
    # Concatenate embeddings
    concat = Concatenate()([fer_processed, ver_processed])
    
    # Dense layers with residual connections
    x = Dense(256, activation='relu')(concat)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Add residual connection
    x_shortcut = Dense(256, activation=None)(concat)
    x = tf.keras.layers.Add()([x, x_shortcut])
    x = Activation('relu')(x)
    
    # Second dense block
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=[fer_input, ver_input], outputs=outputs)
    
    # Compile model with a slightly lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
