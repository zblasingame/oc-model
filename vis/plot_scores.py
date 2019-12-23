import numpy as np
import csv

import plotly.plotly as py
import plotly.graph_objs as go


data = []


for i in range(10):
    with open('../data/mnist/ano_gan/{}/test_set.csv'.format(i), 'r') as f:
        entries = np.array([row for row in csv.reader(f)])
        data.append(np.column_stack((np.ones(entries.shape[0])*i, entries)))

data = np.concatenate(data, axis=0).astype(float)

norm = go.Box(
    y=data[data[:, 1] == 1][:, 2],
    x=data[data[:, 1] == 1][:, 0],
    name='normal',
    marker=dict(
        color='#2ecc71'
    )
)

anom = go.Box(
    y=data[data[:, 1] == 0][:, 2],
    x=data[data[:, 1] == 0][:, 0],
    name='anomalous',
    marker=dict(
        color='#c0392b'
    )
)


layout = go.Layout(
    title='AnoGAN Scores on MNIST Anomaly Detection',
    yaxis=dict(title='Score'),
    boxmode='group'
)

fig = go.Figure(data=(norm, anom), layout=layout)
py.plot(fig, filename='sad/mnist/mnist_ano_gan_scores')
