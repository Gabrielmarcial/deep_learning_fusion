import tensorflow as tf

def conv_block(inputs, filters, kernel_size=3, strides=1, activation='relu', batch_norm=True, dilation_rate = 1):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same', dilation_rate=dilation_rate)(inputs)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    return x

def residual_block(inputs, filters, kernel_size=3, strides=1, activation='relu', batch_norm=True):
    x = conv_block(inputs, filters, kernel_size, strides, activation, batch_norm)
    x = conv_block(x, filters, kernel_size, strides, activation, batch_norm)
    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same')(inputs)
    if batch_norm:
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation(activation)(x)
    return x

def encoder_block(inputs, filters):
    x = residual_block(inputs, filters)
    p = tf.keras.layers.MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, filters):
    x = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(inputs)
    x = tf.keras.layers.concatenate([x, skip_features])
    x = residual_block(x, filters)
    return x

def mlp_block(x, hidden_dim, output_dim):
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
    x = tf.keras.layers.Dense(output_dim)(x)
    return x

def transformer_block(x, num_heads, key_dim, mlp_dim, dropout_rate=0.1):
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)(x, x)
    attn_output = tf.keras.layers.Dropout(dropout_rate)(attn_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    ffn_output = mlp_block(out1, mlp_dim, key_dim)
    ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return out2