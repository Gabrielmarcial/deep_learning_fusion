import tensorflow as tf
from src.utils.blocks import residual_block, encoder_block, decoder_block

#@title ResUNet

def ResUNet(input_shape, n_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bridge
    b1 = residual_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Output
    outputs = tf.keras.layers.Conv2D(n_classes, 1, padding='same', activation='softmax' if n_classes > 1 else 'sigmoid')(d4)

    # Model
    model = tf.keras.models.Model(inputs, outputs, name='ResUNet')
    return model
