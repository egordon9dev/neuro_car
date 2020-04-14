"""
ResNet v1:
[Deep Residual Learning for Image Recognition
](https://arxiv.org/pdf/1512.03385.pdf)
"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, Add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import numpy as np
import os

# Training parameters
def resnet_layer(inputs,
                 num_filters,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same')

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_filters):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 128x72, 8
    stage 1: 64x36, 16
    stage 2:  32x18,  32
    stage 3: 16x9, 64

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 8 != 0:
        raise ValueError('depth should be 8n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_res_blocks = int((depth - 2) / 8)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, num_filters=num_filters)
    # Instantiate the stack of residual units
    for stack in range(4):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = Add()([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    # x = AveragePooling2D(pool_size=(9,16))(x)
    y = Flatten()(x)
    outputs = Dense(1, activation='relu')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def vgg(input_shape, depth, num_filters):
    """
    Features maps sizes:
    stage 0: 128x72, 8
    stage 1: 64x36, 16
    stage 2:  32x18,  32
    stage 3: 16x9, 64

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 8 != 0:
        raise ValueError('depth should be 8n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_blocks = int((depth - 2) / 8)

    inputs = Input(shape=input_shape)
    x = inputs
    # Instantiate the stack of residual units
    for stack in range(4):
        for block in range(num_blocks):
            strides = 1
            if stack > 0 and block == 0:  # first layer but not first stack
                strides = 2  # downsample
            x = Conv2D(filters=num_filters, kernel_size=(3,3),
                             strides=strides,
                             padding="same")(x)
            # x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv2D(filters=num_filters, kernel_size=(3,3),
                             activation="relu",
                             padding="same")(x)
            # x = BatchNormalization()(x)
            x = Activation("relu")(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    # x = AveragePooling2D(pool_size=(9,16))(x)
    y = Flatten()(x)
    outputs = Dense(1, activation='relu')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model