"""Two Class Model

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import numpy as np
import tensorflow as tf
import utils.file_ops
import models.nn


class NN:
    """Two Class NN

    Parameters
    ----------
    layers: list
        List of sizes for each layer including input layer.
        Last layer must be equal to the number of classes.
    activations : function or list, optional
        Activation function or list of activations.
        Note if list then `len(activations) == len(layers) - 1`.
        Default tf.nn.tanh.
    batch_size : int, optional
        Size of mini-batches, default 100.
    n_epochs : int, optional
        Number of training epochs, defailt 100.
    normalize : bool, optional
        Flag to normalize input features, default True.
    normalized_bounds : (a, b) array_like, optional
        Lower, `a`, and upper, `b`, bounds of normalized values
        with `a` < `b`, default (0, 1).
    feature_bounds : (a, b) array_like, optional
        Lower, `a`, and upper, `b`, bounds of dataset values
        with `a` < `b`, default None. Must have same shape as input data.
    l_rate : float, optional
        Learning rate, default 1e-3.
    reg_param : float or bool, optional
        L2 regularization parameter, default 1e-2.
        If False then no regularization is applied.
    debug : bool, optional
        Flag to display debug information, default False.
    display_step : int, optional
        How often to display training debug information, default 10.
    save_path : str, optional
        Where to save model files, default '.models/model.ckpt'
    """

    def __init__(self, layers, **kwargs):
        defaults = {
            'activations': tf.nn.tanh,
            'batch_size': 100,
            'n_epochs': 100,
            'normalize': True,
            'normalized_bounds': (0, 1),
            'feature_bounds': None,
            'l_rate': 1e-3,
            'reg_param': 1e-2,
            'debug': False,
            'display_step': 10,
            'save_path': '.models/model.ckpt'
        }

        self.layers = layers
        n_features = layers[0]
        self.n_features = n_features
        n_classes = layers[-1]

        assert set(kwargs.keys()) <= set(defaults.keys()), (
            'Invalid keyword argument'
        )

        vars(self).update({p: kwargs.get(p, d) for p, d in defaults.items()})

        # Model
        self.X = tf.placeholder(tf.float32, [None, n_features])
        self.Y = tf.placeholder(tf.int64, [None])

        # Normalization parameters
        self.feature_min = tf.Variable(np.zeros(n_features), dtype=tf.float32)
        self.feature_max = tf.Variable(np.zeros(n_features), dtype=tf.float32)

        model = models.nn.gen_fc_neural_net(
            self.layers,
            self.activations,
            1e-2 if not self.reg_param else self.reg_param
        )

        with tf.variable_scope('model'):
            logits = model(self.X)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self.Y
            )
        )

        if self.reg_param is not False:
            self.loss += tf.add_n(tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES
            ))

        self.opt = tf.train.AdamOptimizer(self.l_rate).minimize(self.loss)

        # Eval metrics
        self.pred_labels = tf.argmax(logits, 1)
        self.confusion_matrix = tf.confusion_matrix(
            self.Y,
            self.pred_labels,
            num_classes=n_classes
        )

        self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(
            self.pred_labels,
            self.Y
        )))

        # Variable ops
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def train(self, X, Y, low=None, high=None):
        """Train the model.

        Parameters
        ----------
        X : array_like.
            Features with shape (n_samples, n_features).
        Y : array_like.
            Labels with shape (n_samples).
        """

        if self.normalize:
            _min = X.min(axis=0)
            _max = X.max(axis=0)
            # a, b = self.normalized_bounds
            # a = -1
            # b = 1
            # X = utils.file_ops.rescale(X, _min, _max, a, b)

            # X = np.tanh((X - X.mean(axis=0)) / (_max - _min))

            a = -0.5
            b = 0.5
            alpha = (np.arctanh(b) - np.arctan(a)) / (_max - _min)
            beta = np.arctanh(b) - (alpha * _max)
            X = np.tanh(alpha * X + beta)

        with tf.Session() as sess:
            sess.run(self.init_op)

            batch = utils.file_ops.random_batcher([Y, X], self.batch_size)

            if self.normalize:
                # low, high = self.normalized_bounds
                low, high = -.5, .5
                sess.run(self.feature_min.assign(_min))
                sess.run(self.feature_max.assign(_max))

            elif not self.feature_bounds:
                alpha = 1

                a = X.min(axis=0)
                b = X.max(axis=0)

                low = (0.5 * ((2 + alpha) * a - (alpha * b)))
                high = (0.5 * ((2 + alpha) * b - (alpha * a)))

            else:
                low, high = self.feature_bounds

            for epoch in range(self.n_epochs):
                batch_y, batch_x = next(batch)

                _, loss = sess.run([self.opt, self.loss], feed_dict={
                    self.X: batch_x, self.Y: batch_y
                })

                if not epoch % self.display_step:
                    self.print((
                        'Epoch {0:04} | Loss {1:20.5f}'
                    ).format(epoch+1, loss))

            self.print('Finished training')

            # save model
            save_path = self.saver.save(
                sess,
                self.save_path
            )

            self.print('Model saved in file: {}'.format(save_path))

    def test(self, X, Y):
        """Evaluate model performance.

        Parameters
        ----------
        X : array_like.
            Features with shape (n_samples, n_features).
        Y : array_like.
            Labels with shape (n_samples).

        Returns
        -------
        accuracy : float
            Classification accuracy of model.
        c_mat : np.ndarray
            Confusion matrix.
        """

        with tf.Session() as sess:
            self.saver.restore(sess, self.save_path)

            # normalize data
            if self.normalize:
                _min = self.feature_min.eval()
                _max = self.feature_max.eval()
                # a, b = self.normalized_bounds
                # a = -1
                # b = 1
                # X = utils.file_ops.rescale(X, _min, _max, a, b)

                # X = np.tanh((X - X.mean(axis=0)) / (_max - _min))

                a = -0.5
                b = 0.5
                alpha = (np.arctanh(b) - np.arctan(a)) / (_max - _min)
                beta = np.arctanh(b) - (alpha * _max)
                X = np.tanh(alpha * X + beta)

            acc, mat = sess.run(
                [self.accuracy, self.confusion_matrix],
                feed_dict={
                    self.X: X,
                    self.Y: Y
                }
            )

            self.print('Accuracy = {:.3f}%'.format(acc * 100))
            self.print(mat)

            return acc * 100, mat

    def print(self, val):
        if self.debug:
            print(val)
