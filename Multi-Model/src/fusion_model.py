import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Concatenate, Input, Activation, Add
from src.config import NUM_CLASSES, EMBEDDING_DIM

def build_fusion_model():
    """
    Build the multi-modal fusion model that combines FER and VER embeddings
    with dense blocks and residual connections.
    """
    # Input layers for embeddings
    fer_input = Input(shape=(EMBEDDING_DIM,), name='fer_embedding_input')
    ver_input = Input(shape=(EMBEDDING_DIM,), name='ver_embedding_input')
    
    # Process each modality separately
    fer_processed = Dense(64, activation='relu')(fer_input)
    fer_processed = BatchNormalization()(fer_processed)
    
    ver_processed = Dense(64, activation='relu')(ver_input)
    ver_processed = BatchNormalization()(ver_processed)
    
    # Concatenate the processed embeddings
    concat = Concatenate()([fer_processed, ver_processed])
    
    # First dense block with residual connection
    x = Dense(256, activation='relu')(concat)
    x = BatchNormalization()(x)
    shortcut = Dense(256, activation=None)(concat)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # Second dense block
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Final classification layer
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=[fer_input, ver_input], outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
