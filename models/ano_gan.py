"""AnoGAN Implementation

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import numpy as np
import tensorflow as tf
import utils.file_ops
import models.nn

from tensorflow.keras import layers


class AnoGAN:
    """AnoGAN

    Parameters
    ----------
    batch_size : int, optional
        Size of mini-batches, default 100.
    ano_batch_size : int, optional
        Size of mini-batches for scoring op, default 100.
    n_epochs : int, optional
        Number of training epochs, default 100.
    n_score_epochs : int, optional
        Number of scoring epochs, default 500.
    normalize : bool, optional
        Flag to normalize input features, default True.
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

    def __init__(self, **kwargs):
        defaults = {
            'batch_size': 100,
            'ano_batch_size': 100,
            'n_epochs': 100,
            'n_score_epochs': 500,
            'normalize': True,
            'l_rate': 1e-3,
            'reg_param': 1e-2,
            'debug': False,
            'display_step': 10,
            'save_path': '.models/mriodel.ckpt'
        }

        assert set(kwargs.keys()) <= set(defaults.keys()), (
            'Invalid keyword argument'
        )

        vars(self).update({p: kwargs.get(p, d) for p, d in defaults.items()})

        self.reg_param = 1e-2 if not self.reg_param else self.reg_param
        self.normalized_bounds = (-1, 1)
        self.latent_size = 25
        self.latent_bounds = (-1, 1)
        self.l = 0.5

        # Model
        self.X = tf.placeholder(tf.float32, [None, 28, 28])
        self.Y = tf.placeholder(tf.int64, [None])
        self.Z = tf.placeholder(tf.float32, [None, self.latent_size])

        self.ano_z = tf.get_variable(
            'z',
            [self.ano_batch_size, self.latent_size],
            initializer=tf.random_uniform_initializer(
                minval=self.latent_bounds[0],
                maxval=self.latent_bounds[1]
            )
        )

        self.reseed_ano_z = tf.initializers.variables([self.ano_z])

        # Normalization parameters
        self.feature_min = tf.Variable(np.zeros((28, 28)), dtype=tf.float32)
        self.feature_max = tf.Variable(np.zeros((28, 28)), dtype=tf.float32)

        def discriminator(x):
            # network = [
            #     layers.Reshape((12, 12, 1)),
            #     layers.Conv2D(depth, 3, strides=2, padding='same', activation=tf.nn.leaky_relu),
            #     layers.Conv2D(depth*2, 3, strides=2, padding='same', activation=tf.nn.leaky_relu),
            #     layers.Flatten(),
            #     layers.Dense(2)
            # ]

            network = [
                layers.Reshape((28, 28, 1)),
                layers.Conv2D(16, 3, strides=2, padding='same', activation=tf.nn.leaky_relu),
                layers.Conv2D(32, 3, strides=2, padding='same', activation=tf.nn.leaky_relu),
                layers.Flatten(),
                layers.Dense(2)
            ]

            ith_layer = x

            for i, l in enumerate(network):
                ith_layer = l(ith_layer)

                if i == len(network) - 2:
                    feature_layer = ith_layer

            return ith_layer, feature_layer

        def generator(x):
            # network = [
            #     layers.Dense(3 * 3 * depth*4, activation=tf.nn.leaky_relu),
            #     layers.Reshape((3, 3, depth*4)),
            #     layers.UpSampling2D(),
            #     layers.Conv2DTranspose(depth*2, 3, padding='same'),
            #     layers.Activation('relu'),
            #     layers.UpSampling2D(),
            #     layers.Conv2DTranspose(depth*1, 3, padding='same'),
            #     layers.Activation('relu'),
            #     layers.Conv2DTranspose(1, 3, padding='same'),
            #     layers.Activation('tanh'),
            #     layers.Reshape((12, 12))
            # ]

            network = [
                layers.Dense(7 * 7 * 64, activation=tf.nn.leaky_relu),
                layers.Reshape((7, 7, 64)),
                layers.UpSampling2D(),
                layers.Conv2DTranspose(32, 3, padding='same'),
                layers.Activation('relu'),
                layers.UpSampling2D(),
                layers.Conv2DTranspose(16, 3, padding='same'),
                layers.Activation('relu'),
                layers.Conv2DTranspose(1, 3, padding='same'),
                layers.Activation('sigmoid'),
                layers.Reshape((28, 28))
            ]

            ith_layer = x

            for l in network:
                ith_layer = l(ith_layer)

            return ith_layer

        # pad_X = tf.pad(self.X, ((0, 0), (0, 9), (0, 0)))
        pad_X = self.X

        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            g_X = generator(self.Z)
            g_X_ano = generator(self.ano_z)

        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            d_real, f_real = discriminator(pad_X)
            d_fake, f_fake = discriminator(g_X)
            _, f_fake_ano = discriminator(g_X_ano)

        smooth = 0.1
        d_labels_real = tf.ones_like(d_real) * (1 - smooth)
        d_labels_fake = tf.zeros_like(d_fake)

        with tf.variable_scope('d_loss'):
            d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=d_labels_real,
                logits=d_real
            )
            d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=d_labels_fake,
                logits=d_fake
            )

            self.d_loss = tf.reduce_mean(d_loss_fake + d_loss_real)

        with tf.variable_scope('g_loss'):
            self.g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(d_fake),
                    logits=d_fake
                )
            )

        self.residual_loss = tf.reduce_mean(tf.abs(pad_X - g_X_ano), axis=(1,2))
        self.feature_loss = tf.reduce_mean(tf.abs(f_real - f_fake_ano), axis=1)
        self.anomaly_scores = (
            (1 - self.l) * self.residual_loss + self.l * self.feature_loss
        )

        theta_d = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='discriminator'
        )

        theta_g = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='generator'
        )

        self.d_opt = tf.train.AdamOptimizer(self.l_rate).minimize(
            self.d_loss,
            var_list=theta_d
        )

        self.g_opt = tf.train.AdamOptimizer(self.l_rate).minimize(
            self.g_loss,
            var_list=theta_g
        )

        self.z_opt = tf.train.AdamOptimizer(self.l_rate).minimize(
            self.anomaly_scores,
            var_list=[var for var in tf.trainable_variables() if 'ano_z']
        )

        # Variable ops
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        self.merged = tf.summary.merge_all()

    def train(self, X):
        """Train the GAN on normal samples

        Parameters
        ----------
        X : array_like.
            Features with shape (n_samples, n_features).
        """

        if self.normalize:
            _min = X.min(axis=0)
            _max = X.max(axis=0)
            a, b = self.normalized_bounds
            X = utils.file_ops.rescale(X, _min, _max, a, b)
            # X = utils.file_ops.rescale(X, 0, 1, a, b)

        with tf.Session() as sess:
            sess.run(self.init_op)

            batch = utils.file_ops.random_batcher([X], self.batch_size)

            for epoch in range(self.n_epochs):
                batch_x, = next(batch)

                batch_z = np.random.uniform(
                    self.latent_bounds[0],
                    self.latent_bounds[1],
                    size=(batch_x.shape[0], self.latent_size)
                )

                _, d_loss = sess.run([self.d_opt, self.d_loss], feed_dict={
                    self.X: batch_x,
                    self.Z: batch_z
                })

                batch_z = np.random.uniform(
                    self.latent_bounds[0],
                    self.latent_bounds[1],
                    size=(batch_x.shape[0], self.latent_size)
                )

                _, g_loss = sess.run([self.g_opt, self.g_loss], feed_dict={
                    self.X: batch_x,
                    self.Z: batch_z
                })

                if not epoch % self.display_step:
                    self.print((
                        'Epoch {0:04} | D Loss {1:20.5f} | G Loss {1:20.5f}'
                    ).format(epoch+1, d_loss, g_loss))

            self.print('Finished training')

            if self.normalize:
                sess.run(self.feature_min.assign(_min))
                sess.run(self.feature_max.assign(_max))

            # save model
            save_path = self.saver.save(
                sess,
                self.save_path
            )

            self.print('Model saved in file: {}'.format(save_path))

    def score(self, X):
        """Score samples.

        Parameters
        ----------
        X : array_like
            Features with shape (n_samples, n_features).

        Returns
        -------
        scores : array_like
            Scores of each sample.
        """

        scores = []

        with tf.Session() as sess:
            self.saver.restore(sess, self.save_path)

            # normalize data
            if self.normalize:
                _min = self.feature_min.eval()
                _max = self.feature_max.eval()
                a, b = self.normalized_bounds
                X = utils.file_ops.rescale(X, _min, _max, a, b)
                # X = utils.file_ops.rescale(X, 0, 1, a, b)

            batch = utils.file_ops.batcher([X], self.ano_batch_size)

            for batch_x, in batch:
                if batch_x.shape[0] < self.ano_batch_size:
                    x = np.zeros((self.ano_batch_size, 28, 28))
                    x[:batch_x.shape[0]] = batch_x
                else:
                    x = batch_x

                sess.run(self.reseed_ano_z)

                for epoch in range(self.n_score_epochs):
                    _, batch_scores = sess.run(
                        [self.z_opt, self.anomaly_scores],
                        feed_dict={self.X: x}
                    )

                if batch_x.shape[0] < self.ano_batch_size:
                    batch_scores = batch_scores[:batch_x.shape[0]]

                scores.append(batch_scores)

        return np.concatenate(scores, axis=0)

    def print(self, val):
        if self.debug:
            print(val)
