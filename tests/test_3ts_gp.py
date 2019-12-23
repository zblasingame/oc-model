import sys
import os

main_dir = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(main_dir)

import tensorflow as tf
import numpy as np
import models.amod
import utils.file_ops

import csv

# Model creation
n_features = 40

leak = tf.Variable(.2, name='leak')

l_size = 50
depth = 1

model = models.amod.AMOD(
    # layers=[n_features] + [l_size] * depth + [2],
    layers=[n_features, 100, 50, 2],
    activations=lambda x: tf.nn.leaky_relu(x, 0.2),
    # activations=tf.nn.tanh,
    batch_size=1000,
    n_epochs=1000,
    debug=True,
    l_rate=1e-3,
    reg_param=False,
    normalize=True,
    normalized_bounds=(-1, 1),
    anom_idx=0,
    alpha=1,
    save_path='{}/.models/model.ckpt'.format(main_dir)
)

exploits = [
    'freak', 'poodle', 'nginx_keyleak', 'nginx_rootdir', 'logjam',
    'orzhttpd_rootdir', 'orzhttpd_restore'
]

data = []

for exploit in exploits:
    X, Y = utils.file_ops.load_data(
        '{}/data/raid/gp/{}_data.csv'.format(
            main_dir, exploit
        )
    )

    X1, _ = utils.file_ops.load_data(
        '{}/data/raid/gp/{}_oc_nf.csv'.format(
            main_dir, exploit
        )
    )

    X = np.concatenate((X, X1), axis=1)

    X1 = X[Y == 1]
    Y[Y == -1] = 0
    X0 = X[Y == 0]

    for i in range(5):
        idx = np.random.choice(X1.shape[0], X1.shape[0]//2)
        X_train = X1[idx]
        Y_train = Y[Y == 1][idx]
        model.train(X_train, Y_train)

        for j in range(5):
            idx = np.random.choice(X1.shape[0], X0.shape[0])
            X_test = X1[idx]
            Y_test = Y[Y == 1][idx]

            X_test = np.concatenate((X_test, X0), axis=0)
            Y_test = np.hstack((Y_test, Y[Y == 0]))

            acc, _ = model.test(X_test, Y_test)
            data.append([exploit, acc])


with open('{}/results/oc-cat-3ts-gp.csv'.format(main_dir), 'w') as f:
    writer = csv.writer(f)
    for row in data:
        writer.writerow(row)
