"""AAE Implementation

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import numpy as np
import tensorflow as tf
import importlib
import utils.file_ops


def gen_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)

        return ema_var if ema_var else var

    return ema_getter


class AAE:
    """AAE

    Parameters
    ----------
    dataset : str
        Name of dataset. Used to pull network configuration.
    batch_size : int, optional
        Size of mini-batches, default 100.
    n_epochs : int, optional
        Number of training epochs, default 100.
    degree : int, optional
        Degree of normalization on f_layer scores, default 1.
    debug : bool, optional
        Flag to display debug information, default False.
    display_step : int, optional
        How often to display training debug information, default 10.
    save_path : str, optional
        Where to save model files, default '.models/alad_model.ckpt'
    """

    def __init__(self, dataset, **kwargs):
        defaults = {
            'batch_size': 100,
            'n_epochs': 100,
            'debug': False,
            'degree': 1,
            'display_step': 10,
            'save_path': '.models/alad_model.ckpt'
        }

        assert set(kwargs.keys()) <= set(defaults.keys()), (
            'Invalid keyword argument'
        )

        vars(self).update({p: kwargs.get(p, d) for p, d in defaults.items()})

        # Import model
        model = importlib.import_module('models.aae.{}_model'.format(dataset))

        # Parameters
        self.normalized_bounds = model.NORM_BOUNDS
        self.latent_bounds = model.Z_BOUNDS
        self.normalize = model.NORMALIZE
        self.latent_size = model.LATENT_SIZE

        self.X = tf.placeholder(tf.float32, model.X_SHAPE)
        self.Z = tf.placeholder(tf.float32, [None, model.LATENT_SIZE])
        self.is_training = tf.placeholder(tf.bool)

        # Normalization parameters
        self.feature_min = tf.Variable(
            np.zeros(model.X_SHAPE[1:]), dtype=tf.float32
        )
        self.feature_max = tf.Variable(
            np.zeros(model.X_SHAPE[1:]), dtype=tf.float32
        )

        # Define model
        gen = model.decoder
        enc = model.encoder
        dis = model.discriminator

        with tf.variable_scope('autoencoder_model'):
            with tf.variable_scope('encoder_model'):
                z_gen = enc(self.X, is_training=self.is_training)

            with tf.variable_scope('generator_model'):
                rec_x = gen(z_gen, is_training=self.is_training, reuse=True)

        with tf.variable_scope('discriminator_model'):
            z_logit_real, f_layer = dis(
                self.Z, is_training=self.is_training
            )

            z_logit_fake, f_layer = dis(
                z_gen, is_training=self.is_training, reuse=True
            )

        with tf.name_scope('loss_functions'):
            # Discriminator loss
            z_real_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(z_logit_real),
                logits=z_logit_real
            )
            z_fake_dis = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(z_logit_fake),
                logits=z_logit_fake
            )
            self.discriminator_loss = tf.reduce_mean(z_real_dis + z_fake_dis)

            # gen and encoder
            self.ae_loss = tf.reduce_mean(tf.square(rec_x - self.X))

        with tf.name_scope('optimizers'):
            tvars = tf.trainable_variables()

            d_vars = [v for v in tvars if 'discriminator_model' in v.name]
            g_vars = [v for v in tvars if 'generator_model' in v.name]
            e_vars = [v for v in tvars if 'encoder_model' in v.name]

            u_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            u_ops_d = [x for x in u_ops if 'discriminator_model' in x.name]
            u_ops_g = [x for x in u_ops if 'generator_model' in x.name]
            u_ops_e = [x for x in u_ops if 'encoder_model' in x.name]

            opt = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5)

            with tf.control_dependencies(u_ops_g):
                self.gen_op = opt.minimize(self.ae_loss, var_list=g_vars)

            with tf.control_dependencies(u_ops_e):
                self.enc_op = opt.minimize(self.ae_loss, var_list=e_vars)

            with tf.control_dependencies(u_ops_d):
                self.d_op = opt.minimize(self.discriminator_loss, var_list=d_vars)

            # EMA stuff
            def train_op_ema(vs, op):
                ema = tf.train.ExponentialMovingAverage(decay=0.999)
                maintain_averages_op = ema.apply(vs)

                with tf.control_dependencies([op]):
                    train_op = tf.group(maintain_averages_op)

                return train_op, ema

            self.train_gen_op, gen_ema = train_op_ema(g_vars, self.gen_op)
            self.train_enc_op, enc_ema = train_op_ema(e_vars, self.enc_op)
            self.train_dis_op, dis_ema = train_op_ema(d_vars, self.d_op)

        with tf.variable_scope('encoder_model'):
            z_gen_ema = enc(
                self.X,
                is_training=self.is_training,
                getter=gen_getter(enc_ema),
                reuse=True
            )

        # with tf.variable_scope('generator_model'):
        #     rec_x_ema = gen(
        #         z_gen_ema,
        #         is_training=self.is_training,
        #         getter=gen_getter(gen_ema),
        #         reuse=True
        #     )

        with tf.variable_scope('discriminator_model'):
            _, f_layer_fake_ema = dis(
                z_gen_ema,
                is_training=self.is_training,
                getter=gen_getter(dis_ema),
                reuse=True
            )

        # Scores
        with tf.variable_scope('scores'):
            scores = tf.keras.layers.Flatten()(f_layer_fake_ema)
            self.scores = tf.squeeze(tf.norm(
                scores,
                axis=1,
                ord=self.degree
            ))

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

                batch_z = np.random.normal(
                    # self.latent_bounds[0],
                    # self.latent_bounds[1],
                    size=(batch_x.shape[0], self.latent_size)
                )

                _, ld = sess.run(
                    [
                        self.train_dis_op,
                        self.discriminator_loss
                    ],
                    feed_dict={
                        self.X: batch_x,
                        self.Z: batch_z,
                        self.is_training: True
                    }
                )

                batch_z = np.random.normal(
                    size=(batch_x.shape[0], self.latent_size)
                )

                _, _, la = sess.run(
                    [
                        self.train_gen_op,
                        self.train_enc_op,
                        self.ae_loss,
                    ],
                    feed_dict={
                        self.X: batch_x,
                        self.Z: batch_z,
                        self.is_training: True
                    }
                )

                if not epoch % self.display_step:
                    self.print((
                        'Epoch {0:04} | AE Loss {1:10.5f} | D Loss {1:10.5f}'
                    ).format(epoch+1, la, ld))

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
            Features from data space.

        Returns
        -------
        scores : array_like
            Scores of each sample.
        """

        with tf.Session() as sess:
            self.saver.restore(sess, self.save_path)

            if self.normalize:
                _min = self.feature_min.eval()
                _max = self.feature_max.eval()
                a, b = self.normalized_bounds
                X = utils.file_ops.rescale(X, _min, _max, a, b)

            scores = sess.run(self.scores, feed_dict={
                self.X: X,
                self.Z: np.random.normal(size=[X.shape[0], self.latent_size]),
                self.is_training: False
            })

            return scores

    def print(self, val):
        if self.debug:
            print(val)
