import numpy as np
import csv

import plotly.plotly as py
import plotly.graph_objs as go

exploits = [
    'freak', 'poodle', 'nginx_keyleak', 'nginx_rootdir', 'logjam',
    'orzhttpd_rootdir', 'orzhttpd_restore'
]

ex_map = []

data = []

for i, exploit in enumerate(exploits):
    ex_map.append(exploit)
    ex_data = []
    for j in range(5):
        with open('../data/raid/alad/{}/subset_{}/test_set.csv'.format(
            exploit, j
        ), 'r') as f:
            entries = np.array([row for row in csv.reader(f)])
            ex_data.append(np.column_stack((
                np.ones(entries.shape[0])*i,
                entries
            )))

    ex_data = np.concatenate(ex_data, axis=0).astype(float)
    scores = ex_data[:, 2]
    labels = ex_data[:, 1]
    _min = np.min(scores[labels == 1])
    _max = np.max(scores[labels == 1])
    diff = _max - _min
    a = np.arctanh(-.5)
    b = np.arctanh(.5)
    alpha = (b-a)/diff
    beta = b - (alpha * _max)
    norm_scores = np.tanh(alpha * scores + beta)

    ex_data[:, 2] = norm_scores

    data.append(ex_data)

data = np.concatenate(data, axis=0)

norm = go.Box(
    y=data[data[:, 1] == 1][:, 2],
    x=[ex_map[int(i)] for i in data[data[:, 1] == 1][:, 0]],
    name='normal',
    marker=dict(
        color='#2ecc71'
    )
)

anom = go.Box(
    y=data[data[:, 1] == 0][:, 2],
    x=[ex_map[int(i)] for i in data[data[:, 1] == 0][:, 0]],
    name='anomalous',
    marker=dict(
        color='#c0392b'
    )
)


layout = go.Layout(
    title='ALAD Scores on 3 Time Step Anomaly Detection Normalized',
    yaxis=dict(title='Score'),
    boxmode='group'
)

fig = go.Figure(data=(norm, anom), layout=layout)
py.plot(fig, filename='sad/mnist/3ts_alad_scores_norm')
