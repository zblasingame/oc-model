"""ALAD Network for 3ts Camel dataset

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


X_SHAPE = (None, 36)
LATENT_SIZE = 5
NORMALIZE = True
NORM_BOUNDS = (-1, 1)
Z_BOUNDS = (-1, 1)


def encoder(x, getter=None, reuse=False, is_training=False):
    """ Encoder network

    Parameters
    ----------
    x : tensor
        Input tensor.
    getter : function, optional
        Function to get variabnles for exponential moving average.
        Default None.
    reuse : bool, optional
        Flag to share variables, default False.
    is_training : bool, optional
        Flag that model is being trained, default False.

    Returns
    -------
    net : tensor
        Last layer of the encoder.
    """

    with tf.variable_scope('encoder', reuse=reuse, custom_getter=getter):
        with tf.variable_scope('layer_1'):
            x = layers.Dense(100)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(0.5)(x, training=is_training)

        with tf.variable_scope('layer_2'):
            x = layers.Dense(50)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(0.5)(x, training=is_training)

        with tf.variable_scope('layer_3'):
            x = layers.Dense(LATENT_SIZE, activation='tanh')(x)

    return x


def decoder(z, getter=None, reuse=False, is_training=False):
    """ Decoder network

    Parameters
    ----------
    z : tensor
        Input tensor.
    getter : function, optional
        Function to get variabnles for exponential moving average.

        Default None.
    reuse : bool, optional
        Flag to share variables, default False.
    is_training : bool, optional
        Flag that model is being trained, default False.

    Returns
    -------
    net : tensor
        Last layer of the encoder.
    """

    with tf.variable_scope('decoder', reuse=reuse, custom_getter=getter):
        with tf.variable_scope('layer_1'):
            z = layers.Dense(50)(z)
            z = layers.LeakyReLU()(z)
            z = layers.Dropout(0.5)(z, training=is_training)

        with tf.variable_scope('layer_2'):
            z = layers.Dense(100)(z)
            z = layers.LeakyReLU()(z)
            z = layers.Dropout(0.5)(z, training=is_training)

        with tf.variable_scope('layer_3'):
            z = layers.Dense(np.prod(X_SHAPE[1:]), activation='tanh')(z)
            z = layers.Reshape(X_SHAPE[1:])(z)

    return z


def discriminator(z, getter=None, reuse=False, is_training=False):
    """ D_z network

    Parameters
    ----------
    z : tensor
        Latent space input tensor.
    getter : function, optional
        Function to get variabnles for exponential moving average.
        Default None.
    reuse : bool, optional
        Flag to share variables, default False.
    is_training : bool, optional
        Flag that model is being trained, default False.

    Returns
    -------
    net : tensor
        Last layer of the encoder.
    """

    with tf.variable_scope(
        'discriminator', reuse=reuse, custom_getter=getter
    ):
        with tf.variable_scope('z_layer_1'):
            z = layers.Dense(200)(z)
            z = layers.LeakyReLU()(z)
            z = layers.Dropout(0.5)(z, training=is_training)

        with tf.variable_scope('z_layer_2'):
            z = layers.Dense(100)(z)
            z = layers.LeakyReLU()(z)
            z = layers.Dropout(0.5)(z, training=is_training)

        with tf.variable_scope('z_layer_3'):
            z = layers.Dense(50)(z)
            z = layers.LeakyReLU()(z)
            z = layers.Dropout(0.5)(z, training=is_training)

        f_layer = z

        with tf.variable_scope('z_layer_4'):
            z = layers.Dense(2)(z)
            logits = tf.squeeze(z)

    return logits, f_layer
