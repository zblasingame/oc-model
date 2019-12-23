import sys
import os

main_dir = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(main_dir)

import tensorflow as tf
import numpy as np
import models.ano_gan
import utils.file_ops

from tqdm import tqdm


# Model creation
model = models.ano_gan.AnoGAN(
    n_epochs=100,
    batch_size=100,
    n_score_epochs=20,
    ano_batch_size=1000,
    normalize=True,
    debug=True,
    save_path='{}/.models/ano_gan.ckpt'.format(main_dir)
)

exploits = [
    'freak', 'poodle', 'nginx_keyleak', 'nginx_rootdir', 'logjam',
    'orzhttpd_rootdir', 'orzhttpd_restore'
]

data = []

for exploit in tqdm(exploits):
    for i in range(5):
        X_train, Y_train = utils.file_ops.load_data(
            '{}/data/raid/ts/{}/subset_{}/train_set.csv'.format(
                main_dir, exploit, i
            )
        )

        X_train = X_train.reshape(-1, 3, 12)

        model.train(X_train)

        X_test, Y_test = utils.file_ops.load_data(
            '{}/data/raid/ts/{}/subset_{}/test_set.csv'.format(
                main_dir, exploit, i
            )
        )


        X_test = X_test.reshape(-1, 3, 12)

        Y_test[Y_test == -1] = 0

        scores_train = model.score(X_train)
        scores_test = model.score(X_test)

        np.savetxt(
            '{}/data/raid/ano_gan_cnn/{}/subset_{}/train_set.csv'.format(
                main_dir, exploit, i
            ),
            np.column_stack((Y_train, scores_train)),
            delimiter=','
        )

        np.savetxt(
            '{}/data/raid/ano_gan_cnn/{}/subset_{}/test_set.csv'.format(
                main_dir, exploit, i
            ),
            np.column_stack((Y_test, scores_test)),
            delimiter=','
        )
