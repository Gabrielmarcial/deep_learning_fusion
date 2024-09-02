import tensorflow as tf
from src.utils.blocks import conv_block, residual_block

def ResNet(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Initial convolution and max-pooling
    x = conv_block(inputs, 64, kernel_size=7, strides=2)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    low_level_features = x

    x = residual_block(x, 128, strides=2)
    x = residual_block(x, 128)

    x = residual_block(x, 256, strides=2)
    x = residual_block(x, 256)

    x = residual_block(x, 512, strides=2)
    x = residual_block(x, 512)

    model = tf.keras.models.Model(inputs, x, name='ResNet')
    return model, low_level_features
