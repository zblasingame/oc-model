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

l_size = 100
depth = 2

model = models.amod.AMOD(
    layers=[n_features] + [l_size] * depth + [2],
    activations=lambda x: tf.nn.leaky_relu(x, leak),
    batch_size=500,
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

exploits = [str(i) for i in range(10)]

data = []

for exploit in exploits:
    X_train, Y_train = utils.file_ops.load_data(
        '{}/data/mnist/ano_gan/{}/train_set.csv'.format(
            main_dir, exploit
        )
    )

    model.train(X_train, Y_train)

    X_test, Y_test = utils.file_ops.load_data(
        '{}/data/mnist/ano_gan/{}/test_set.csv'.format(
            main_dir, exploit
        )
    )

    acc, _ = model.test(X_test, Y_test)
    data.append([exploit, acc])

with open('{}/results/mnist-ano-gan.csv'.format(main_dir), 'w') as f:
    writer = csv.writer(f)
    for row in data:
        writer.writerow(row)
