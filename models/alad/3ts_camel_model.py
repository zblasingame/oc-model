"""ALAD Network for 3ts Camel dataset

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


X_SHAPE = (None, 36)
LATENT_SIZE = 10
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


def discriminator_xz(x, z, getter=None, reuse=False, is_training=False):
    """ D_xz network

    Parameters
    ----------
    x : tensor
        Data space input tensor.
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
        'discriminator_xz', reuse=reuse, custom_getter=getter
    ):
        with tf.variable_scope('x_layer_1'):
            x = layers.Dense(100)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(0.5)(x, training=is_training)

        with tf.variable_scope('x_layer_2'):
            x = layers.Dense(50)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(0.5)(x, training=is_training)

        with tf.variable_scope('x_layer_3'):
            x = layers.Dense(LATENT_SIZE, activation='tanh')(x)

        xz = tf.concat([x, z], axis=1)

        with tf.variable_scope('xz_layer_1'):
            xz = layers.Dense(200)(xz)
            xz = layers.LeakyReLU()(xz)
            xz = layers.Dropout(0.5)(xz, training=is_training)

        with tf.variable_scope('xz_layer_2'):
            xz = layers.Dense(100)(xz)
            xz = layers.LeakyReLU()(xz)
            xz = layers.Dropout(0.5)(xz, training=is_training)

        with tf.variable_scope('xz_layer_3'):
            xz = layers.Dense(50)(xz)
            xz = layers.LeakyReLU()(xz)
            xz = layers.Dropout(0.5)(xz, training=is_training)

        f_layer = xz

        with tf.variable_scope('xz_layer_4'):
            xz = layers.Dense(2)(xz)
            logits = tf.squeeze(xz)

    return logits, f_layer


def discriminator_xx(x1, x2, getter=None, reuse=False, is_training=False):
    """ D_xx network

    Parameters
    ----------
    x1 : tensor
        Data space input tensor.
    x2 : tensor
        Reconstructed tensor.
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
        'discriminator_xx', reuse=reuse, custom_getter=getter
    ):
        xx = tf.concat([x1, x2], axis=1)

        with tf.variable_scope('xx_layer_1'):
            xx = layers.Dense(200)(xx)
            xx = layers.LeakyReLU()(xx)
            xx = layers.Dropout(0.5)(xx, training=is_training)

        with tf.variable_scope('xx_layer_2'):
            xx = layers.Dense(100)(xx)
            xx = layers.LeakyReLU()(xx)
            xx = layers.Dropout(0.5)(xx, training=is_training)

        with tf.variable_scope('xx_layer_3'):
            xx = layers.Dense(50)(xx)
            xx = layers.LeakyReLU()(xx)
            xx = layers.Dropout(0.5)(xx, training=is_training)

        f_layer = xx

        with tf.variable_scope('xx_layer_4'):
            xx = layers.Dense(2)(xx)
            logits = tf.squeeze(xx)

    return logits, f_layer


def discriminator_zz(z1, z2, getter=None, reuse=False, is_training=False):
    """ D_zz network

    Parameters
    ----------
    z1 : tensor
        Latent space input tensor.
    z2 : tensor
        Reconstructed tensor.
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
        'discriminator_zz', reuse=reuse, custom_getter=getter
    ):
        zz = tf.concat([z1, z2], axis=1)

        with tf.variable_scope('zz_layer_1'):
            zz = layers.Dense(200)(zz)
            zz = layers.LeakyReLU()(zz)
            zz = layers.Dropout(0.5)(zz, training=is_training)

        with tf.variable_scope('zz_layer_2'):
            zz = layers.Dense(100)(zz)
            zz = layers.LeakyReLU()(zz)
            zz = layers.Dropout(0.5)(zz, training=is_training)

        with tf.variable_scope('zz_layer_3'):
            zz = layers.Dense(50)(zz)
            zz = layers.LeakyReLU()(zz)
            zz = layers.Dropout(0.5)(zz, training=is_training)

        f_layer = zz

        with tf.variable_scope('zz_layer_4'):
            zz = layers.Dense(2)(zz)
            logits = tf.squeeze(zz)

    return logits, f_layer
