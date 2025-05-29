import tensorflow as tf


def build_unet_model(input_shape=(256, 256, 1)):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder block 1
    c1 = tf.keras.layers.Conv2D(64, 3, padding='same')(inputs)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Activation('relu')(c1)
    c1 = tf.keras.layers.Conv2D(64, 3, padding='same')(c1)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Activation('relu')(c1)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    # Encoder block 2
    c2 = tf.keras.layers.Conv2D(128, 3, padding='same')(p1)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Activation('relu')(c2)
    c2 = tf.keras.layers.Conv2D(128, 3, padding='same')(c2)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Activation('relu')(c2)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    # Encoder block 3
    c3 = tf.keras.layers.Conv2D(256, 3, padding='same')(p2)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Activation('relu')(c3)
    c3 = tf.keras.layers.Conv2D(256, 3, padding='same')(c3)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Activation('relu')(c3)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    # Encoder block 4
    c4 = tf.keras.layers.Conv2D(512, 3, padding='same')(p3)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Activation('relu')(c4)
    c4 = tf.keras.layers.Conv2D(512, 3, padding='same')(c4)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Activation('relu')(c4)
    p4 = tf.keras.layers.MaxPooling2D()(c4)

    # Bottleneck
    c5 = tf.keras.layers.Conv2D(1024, 3, padding='same')(p4)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Activation('relu')(c5)
    c5 = tf.keras.layers.Conv2D(1024, 3, padding='same')(c5)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Activation('relu')(c5)

    # Decoder block 1
    u6 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    u6 = tf.keras.layers.Concatenate()([u6, c4])
    c6 = tf.keras.layers.Conv2D(512, 3, padding='same')(u6)
    c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.Activation('relu')(c6)
    c6 = tf.keras.layers.Conv2D(512, 3, padding='same')(c6)
    c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.Activation('relu')(c6)

    # Decoder block 2
    u7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    u7 = tf.keras.layers.Concatenate()([u7, c3])
    c7 = tf.keras.layers.Conv2D(256, 3, padding='same')(u7)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.Activation('relu')(c7)
    c7 = tf.keras.layers.Conv2D(256, 3, padding='same')(c7)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.Activation('relu')(c7)

    # Decoder block 3
    u8 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    u8 = tf.keras.layers.Concatenate()([u8, c2])
    c8 = tf.keras.layers.Conv2D(128, 3, padding='same')(u8)
    c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.layers.Activation('relu')(c8)
    c8 = tf.keras.layers.Conv2D(128, 3, padding='same')(c8)
    c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.layers.Activation('relu')(c8)

    # Decoder block 4
    u9 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    u9 = tf.keras.layers.Concatenate()([u9, c1])
    c9 = tf.keras.layers.Conv2D(64, 3, padding='same')(u9)
    c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = tf.keras.layers.Activation('relu')(c9)
    c9 = tf.keras.layers.Conv2D(64, 3, padding='same')(c9)
    c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = tf.keras.layers.Activation('relu')(c9)

    # Output layer: 1 filter with sigmoid activation
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(c9)

    model = tf.keras.models.Model(inputs, outputs)

    return model

def load_unet_model(model_path = "models/project.h5"):
    model = build_unet_model()
    model.load_weights(model_path)
    return model