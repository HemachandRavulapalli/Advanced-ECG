import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, initializers

# ------------------------------------
# 1️⃣ Improved 1D CNN for ECG Signals
# ------------------------------------
def build_cnn_1d(input_shape, num_classes, dropout_rate=0.5):
    """
    Enhanced 1D CNN model for ECG classification.
    Includes BatchNorm, L2 regularization, and He initialization.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv1D(64, kernel_size=7, padding='same', activation='relu',
                      kernel_initializer=initializers.HeNormal()),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(128, kernel_size=5, padding='same', activation='relu',
                      kernel_initializer=initializers.HeNormal()),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(256, kernel_size=3, padding='same', activation='relu',
                      kernel_initializer=initializers.HeNormal()),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(dropout_rate),

        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# ------------------------------------
# 2️⃣ Improved 2D CNN for Spectrograms
# ------------------------------------
def build_cnn_2d(input_shape, num_classes, dropout_rate=0.5):
    """
    Enhanced 2D CNN for ECG spectrogram-like representations.
    Uses BatchNorm, L2 regularization, and He initialization.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                      kernel_initializer=initializers.HeNormal()),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                      kernel_initializer=initializers.HeNormal()),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                      kernel_initializer=initializers.HeNormal()),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(dropout_rate),

        layers.Dense(num_classes, activation='softmax')
    ])

    return model
