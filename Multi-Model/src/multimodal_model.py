from tensorflow.keras.layers import concatenate, Dense, Dropout
from fer_model import build_fer_model
from ver_model import build_ver_model
from tensorflow.keras.models import Model

# Load models
fer_model = build_fer_model()
ver_model = build_ver_model()

# Concatenate features
merged = concatenate([fer_model.output, ver_model.output])
x = Dense(128, activation="relu")(merged)
x = Dropout(0.3)(x)
output = Dense(7, activation="softmax")(x)

# Multimodal Model
multimodal_model = Model(inputs=[fer_model.input, ver_model.input], outputs=output)
