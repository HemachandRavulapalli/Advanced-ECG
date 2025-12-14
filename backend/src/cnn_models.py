import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ======================================================
# CNN-1D : Temporal ECG Model
# ======================================================
def build_cnn_1d(input_shape=(1000, 1), num_classes=5, dropout_rate=0.3):
    """
    CNN-1D for ECG rhythm and morphology learning
    Input shape: (1000, 1)
    """

    inputs = layers.Input(shape=input_shape)

    # Multi-scale temporal filters
    x1 = layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x2 = layers.Conv1D(32, 5, padding="same", activation="relu")(inputs)
    x3 = layers.Conv1D(32, 7, padding="same", activation="relu")(inputs)

    x = layers.Concatenate()([x1, x2, x3])
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Residual block
    shortcut = layers.Conv1D(64, 1, padding="same")(x)

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Classifier
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


# ======================================================
# CNN-2D : Reshaped ECG Model
# ======================================================
def build_cnn_2d(input_shape=(100, 10, 1), num_classes=5, dropout_rate=0.3):
    """
    CNN-2D for reshaped ECG representation
    Input shape: (100, 10, 1)
    """

    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model
