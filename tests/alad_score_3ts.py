import sys
import os

main_dir = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(main_dir)

import tensorflow as tf
import numpy as np
import models.alad.model
import utils.file_ops

from tqdm import tqdm


# Model creation
model = models.alad.model.ALAD(
    n_epochs=100,
    batch_size=100,
    dataset='3ts_camel',
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

        model.train(X_train)

        X_test, Y_test = utils.file_ops.load_data(
            '{}/data/raid/ts/{}/subset_{}/test_set.csv'.format(
                main_dir, exploit, i
            )
        )

        Y_test[Y_test == -1] = 0

        scores_train = model.score(X_train)
        scores_test = model.score(X_test)

        np.savetxt(
            '{}/data/raid/alad/{}/subset_{}/train_set.csv'.format(
                main_dir, exploit, i
            ),
            np.column_stack((Y_train, scores_train)),
            delimiter=','
        )

        np.savetxt(
            '{}/data/raid/alad/{}/subset_{}/test_set.csv'.format(
                main_dir, exploit, i
            ),
            np.column_stack((Y_test, scores_test)),
            delimiter=','
        )
