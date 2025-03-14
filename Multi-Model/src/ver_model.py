from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Bidirectional, Dropout

def build_ver_model():
    input_layer = Input(shape=(128, 128, 1))
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(7, activation="softmax")(x)
    return Model(inputs=input_layer, outputs=output, name="VER_Model")
