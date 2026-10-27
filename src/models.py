"""
models.py - CNN, LSTM and CNN-LSTM models for speech emotion recognition.
Input shape: (130 time frames, 80 features)
"""
import keras
from keras import layers


def build_cnn(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)
    x = inp
    for filters in (64, 128, 128):
        x = layers.Conv1D(filters, 5, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.3)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="cnn")


def build_lstm(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=True)(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="lstm")


def build_cnn_lstm(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)
    x = inp
    for filters in (64, 128):
        x = layers.Conv1D(filters, 5, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="cnn_lstm")


MODELS = {
    "cnn": build_cnn,
    "lstm": build_lstm,
    "cnn_lstm": build_cnn_lstm,
}


def get_model(name, input_shape, n_classes):
    """Build and compile a model by name ('cnn', 'lstm' or 'cnn_lstm')."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS)}")
    model = MODELS[name](input_shape, n_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model