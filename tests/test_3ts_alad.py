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
n_features = 1

leak = tf.Variable(.2, name='leak')

l_size = 50
depth = 1

model = models.amod.AMOD(
    # layers=[n_features] + [l_size] * depth + [2],
    layers=[n_features, 50, 2],
    # activations=lambda x: tf.nn.leaky_relu(x, leak),
    activations=tf.nn.tanh,
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
    for i in range(5):
        X_train, Y_train = utils.file_ops.load_data(
            '{}/data/raid/alad/{}/subset_{}/train_set.csv'.format(
                main_dir, exploit, i
            )
        )
        # X_train, Y_train = utils.file_ops.load_data(
        #     '{}/data/raid/ano_gan/{}/subset_{}/test_set.csv'.format(
        #         main_dir, exploit, i
        #     )
        # )

        Y_train[Y_train == -1] = 0
        model.train(X_train, Y_train)

        for j in range(5):
            X_test, Y_test = utils.file_ops.load_data(
                '{}/data/raid/alad/{}/subset_{}/test_set.csv'.format(
                    main_dir, exploit, j
                )
            )

            Y_test[Y_test == -1] = 0

            acc, _ = model.test(X_test, Y_test)
            data.append([exploit, acc])


with open('{}/results/3ts-alad-sat-large.csv'.format(main_dir), 'w') as f:
    writer = csv.writer(f)
    for row in data:
        writer.writerow(row)
