"""
A module for a variety of ML models.

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import tensorflow as tf
import numpy as np


def _build_fc_layer(
    X,
    act,
    input_dim,
    output_dim,
    layer_id,
    reg_param=0.01,
    drop_prob=False
):
    """Builds a fully connected layer.

    Parameters
    ----------
    X : tf.Tensor
        Input tensor with size `input_dim`.
    act : function
        Activation function.
    input_dim : int
        Size of input dimension.
    output_dim : int
        Size of output dimension.
    layer_id : str or int
        ID of layer.
    reg_param : float, optional
        L2 regularization parameter, default=0.01.
    drop_prob : bool or tf.Variable, optional
        Probability of dropout layer. If False the
        dropout layer is not created. Otherwise, the
        value must be of type tf.Variable, default False.

    Returns
    -------
    out : tf.Tensor
        Output of fully connected layer with size `output_dim`.
    """

    # initializer = tf.random_normal_initializer(mean=0, stddev=0.1)
    initializer = tf.contrib.layers.xavier_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(reg_param)

    with tf.variable_scope('fc_layer_{}'.format(layer_id)):
        w1 = tf.get_variable(
            'w1',
            [input_dim, output_dim],
            initializer=initializer,
            regularizer=regularizer
        )
        b1 = tf.get_variable(
            'b1',
            [output_dim],
            initializer=initializer,
            regularizer=regularizer
        )

        if drop_prob is not False:
            return tf.nn.dropout(
                act(tf.add(tf.matmul(X, w1), b1)),
                keep_prob=drop_prob
            )
        else:
            return act(tf.add(tf.matmul(X, w1), b1))


def gen_fc_neural_net(layers, activations, reg_param, drop_prob=False):
    """Creates function that generates fully connected neural net.

    Parameters
    ----------
    layers: list
        List of sizes for each layer including input layer.
    activations : function or list
        Activation function or list of activations.
        Note if list then `len(activations) == len(layers) - 1`.
    reg_param : float, optional
        L2 regularization parameter, default=0.01.
    drop_prob : bool or tf.Variable, optional
        Probability of dropout layer. If False the
        dropout layer is not created. Otherwise, the
        value must be of type tf.Variable, default False.

    Returns
    -------
    fc_neural_net : function
        Function that generates fully connected neural net.

        Parameters
        ----------
        X : tf.Tensors
            Input tensor of shape [batch_size, n_inputs].

        Returns
        -------
        out : tf.Tensor
            Tensor holding the output of the neural net.
    """
    def _fc_neural_net(X):
        ith_layer_out = X

        if type(activations) is list:
            assert len(activations) == len(layers) - 1, (
                'Mismatch between activations and layers'
            )
            for i in range(len(layers) - 1):
                ith_layer_out = _build_fc_layer(
                    ith_layer_out,
                    activations[i],
                    layers[i],
                    layers[i+1],
                    i,
                    reg_param,
                    drop_prob
                )
        else:
            for i in range(len(layers) - 1):
                ith_layer_out = _build_fc_layer(
                    ith_layer_out,
                    activations,
                    layers[i],
                    layers[i+1],
                    i,
                    reg_param,
                    drop_prob
                )

        return ith_layer_out

    return _fc_neural_net
