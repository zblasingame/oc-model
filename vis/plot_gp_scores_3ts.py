import numpy as np
import csv

import plotly.plotly as py
import plotly.graph_objs as go

exploits = [
    'freak', 'poodle', 'nginx_keyleak', 'nginx_rootdir', 'logjam',
    'orzhttpd_rootdir', 'orzhttpd_restore'
]

for exploit in exploits:
    with open('../data/raid/gp/{}_nf.csv'.format(exploit), 'r') as f:
        next(csv.reader(f))
        data = np.array([row for row in csv.reader(f)]).astype(float)

    Y = data[:, 0]
    X = data[:, 1:]

    n_features = X.shape[1]

    norm = go.Box(
        y=X[Y==1].flatten(),
        x=np.repeat(['F1', 'F2', 'F3', 'F4'], n_features),
        name='normal',
        marker=dict(
            color='#2ecc71'
        )
    )

    anom = go.Box(
        y=X[Y==-1].flatten(),
        x=np.repeat(['F1', 'F2', 'F3', 'F4'], n_features),
        name='anomalous',
        marker=dict(
            color='#c0392b'
        )
    )

    layout = go.Layout(
        title='GP Features on Distribution for {}'.format(exploit),
        yaxis=dict(title='Value'),
        boxmode='group'
    )

    fig = go.Figure(data=(norm, anom), layout=layout)
    py.plot(fig, filename='sad/gp/gp_{}_box'.format(exploit), auto_open=False)
