import tensorflow as tf
from src.utils.blocks import conv_block
from src.models.resnet import ResNet

#@title DeepLabv3plus

def atrous_spatial_pyramid_pooling(x):
    dims = x.shape
    pool1 = tf.keras.layers.AveragePooling2D(pool_size=(dims[1], dims[2]))(x)
    pool1 = tf.keras.layers.Conv2D(256, 1, padding='same')(pool1)
    pool1 = tf.keras.layers.BatchNormalization()(pool1)
    pool1 = tf.keras.layers.Activation('relu')(pool1)
    pool1 = tf.keras.layers.UpSampling2D((dims[1], dims[2]), interpolation='bilinear')(pool1)
    conv1 = conv_block(x, 256, 1, dilation_rate = 1)
    conv2 = conv_block(x, 256, 3, dilation_rate = 6)
    conv3 = conv_block(x, 256, 3, dilation_rate = 12)
    conv4 = conv_block(x, 256, 3, dilation_rate = 18)
    x = tf.keras.layers.concatenate([pool1, conv1, conv2, conv3, conv4])
    x = tf.keras.layers.Conv2D(256, 1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def DeepLabV3Plus(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Backbone alternative
    #base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    #image_features = base_model.get_layer('conv4_block6_2_relu').output
    #low_level_features = base_model.get_layer('conv2_block3_2_relu').output

    # Backbone current
    image_features, low_level_features = ResNet(input_shape)
    x = image_features(inputs)

    # Atrous Spatial Pyramid Pooling
    x_a = atrous_spatial_pyramid_pooling(x)

    # Decoder
    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x_a)

    low_level_features = tf.keras.layers.Conv2D(48, 1, padding='same')(low_level_features)
    low_level_features = tf.keras.layers.BatchNormalization()(low_level_features)
    low_level_features = tf.keras.layers.Activation('relu')(low_level_features)

    x = tf.keras.layers.concatenate([x, low_level_features])
    x = conv_block(x, 256, 3)
    x = conv_block(x, 256, 3)

    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2D(num_classes, 1, padding='same')(x)
    x = tf.keras.layers.Activation('softmax' if num_classes > 1 else 'sigmoid')(x)

    model = tf.keras.models.Model(inputs, x, name='DeepLabV3Plus')
    return model

