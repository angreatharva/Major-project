import tensorflow as tf

def convert_to_tflite():
    # Load the trained Keras model
    model = tf.keras.models.load_model('./models/emotion_model.keras')  # Path to your saved model

    # Create a TFLiteConverter instance to convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Perform the conversion
    tflite_model = converter.convert()

    # Save the converted TFLite model to a file
    with open('./models/emotion_model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Model successfully converted to TFLite format.")

if __name__ == "__main__":
    convert_to_tflite()
