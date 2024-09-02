import tensorflow as tf
from src.utils.blocks import transformer_block, encoder_block, decoder_block

#@title UNetR

def transformer_encoder(x, num_layers, num_heads, key_dim, mlp_dim, dropout_rate=0.1):
    for _ in range(num_layers):
        x = transformer_block(x, num_heads, key_dim, mlp_dim, dropout_rate)
    return x

def UNETR(input_shape, num_classes, num_transformer_layers, num_heads, key_dim, mlp_dim, dropout_rate=0.1):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder (ResNet-like)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Transformer encoder
    p4_shape = p4.shape
    p4_reshaped = tf.keras.layers.Reshape((-1, p4_shape[-1]))(p4)
    p4_transformed = transformer_encoder(p4_reshaped, num_transformer_layers, num_heads, key_dim, mlp_dim, dropout_rate)
    p4_transformed = tf.keras.layers.Reshape(p4_shape[1:])(p4_transformed)

    # Decoder
    d1 = decoder_block(p4_transformed, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Output
    outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same', activation='softmax' if num_classes > 1 else 'sigmoid')(d4)

    # Model
    model = tf.keras.models.Model(inputs, outputs, name='UNETR')
    return model
