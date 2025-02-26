
### main.py
import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from scripts.train_model import create_emotion_model, create_transfer_learning_model, compile_model
from scripts.preprocess_data import load_and_preprocess_data

def train_and_evaluate():
    BATCH_SIZE = 32
    EPOCHS = 50
    MODEL_PATH_CNN = 'models/emotion_model.keras'
    MODEL_PATH_TL = 'models/emotion_transfer_model.keras'
    DATASET_PATH = 'dataset/'
    os.makedirs('models', exist_ok=True)
    train_dataset, val_dataset = load_and_preprocess_data(DATASET_PATH, BATCH_SIZE)
    classes = np.unique(train_dataset.classes)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_dataset.classes)
    class_weight_dict = dict(zip(classes, class_weights))
    cnn_model = create_emotion_model()
    transfer_model = create_transfer_learning_model()
    compile_model(cnn_model)
    compile_model(transfer_model)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(MODEL_PATH_CNN, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)]
    cnn_model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=callbacks, class_weight=class_weight_dict, verbose=1)
    transfer_model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=[tf.keras.callbacks.ModelCheckpoint(MODEL_PATH_TL, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)], verbose=1)

def main():
    train_and_evaluate()

if __name__ == "__main__":
    main()