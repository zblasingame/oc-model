import sys
import os

main_dir = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(main_dir)

import tensorflow as tf
import numpy as np
import models.ano_gan
import utils.file_ops

from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/')


# Model creation
model = models.ano_gan.AnoGAN(
    # n_features=784,
    n_epochs=100,
    batch_size=100,
    n_score_epochs=20,
    ano_batch_size=50,
    normalize=False,
    debug=True,
    save_path='{}/.models/ano_gan.ckpt'.format(main_dir)
)

# datasets = {
#     'even': [0,2,4,6,8],
#     'only-0': [0],
#     'all-but-0': [1,2,3,4,5,6,7,8,9],
#     'mod-3': [3,6,9],
#     'less-5': [0,1,2,3,4],
# }

datasets = {str(i): [i] for i in range(10)}

data = []

for dataset, norm_keys in tqdm(datasets.items()):
    X_train, Y_train = mnist.train.images, mnist.train.labels.copy()
    X_test, Y_test = mnist.test.images, mnist.test.labels.copy()

    idx = np.in1d(Y_train, norm_keys)
    Y_train[idx] = 1

    X_train = X_train[idx].reshape(-1, 28, 28)
    X_test = X_test.reshape(-1, 28, 28)
    Y_train = Y_train[idx]

    idx = np.in1d(Y_test, norm_keys)
    Y_test[idx] = 1
    Y_test[idx == 0] = 0

    model.train(X_train)

    scores_train = model.score(X_train)
    scores_test = model.score(X_test)

    np.savetxt(
        '{}/data/mnist/ano_gan/{}/train_set.csv'.format(
            main_dir, dataset
        ),
        np.column_stack((Y_train, scores_train)),
        delimiter=','
    )

    np.savetxt(
        '{}/data/mnist/ano_gan/{}/test_set.csv'.format(
            main_dir, dataset
        ),
        np.column_stack((Y_test, scores_test)),
        delimiter=','
    )
