import numpy as np
import csv

import plotly.plotly as py
import plotly.graph_objs as go


with open('../results/3ts-ano-gan-no-norm.csv', 'r') as f:
    data_0 = np.array([row for row in csv.reader(f)])

with open('../results/3ts-ano-gan-norm.csv', 'r') as f:
    data_1 = np.array([row for row in csv.reader(f)])

with open('../results/3ts-ano-gan-sat.csv', 'r') as f:
    data_2 = np.array([row for row in csv.reader(f)])

with open('../results/3ts-ano-gan-sat-large.csv', 'r') as f:
    data_3 = np.array([row for row in csv.reader(f)])

boxes = [go.Box(
    y=data_0[:, 1].astype(np.float),
    x=data_0[:, 0],
    name='No Norm'
),
go.Box(
    y=data_1[:, 1].astype(np.float),
    x=data_1[:, 0],
    name='Norm'
),
go.Box(
    y=data_2[:, 1].astype(np.float),
    x=data_2[:, 0],
    name='Saturator'
),
go.Box(
    y=data_1[:, 1].astype(np.float),
    x=data_1[:, 0],
    name='Saturator Large Sample Range'
)
]


layout = go.Layout(
    title='Performance of AnoGAN CNN Scores on 3ts Dataset',
    yaxis=dict(title='Accuracy (%)'),
    boxmode='group'
)

fig = go.Figure(data=boxes, layout=layout)
py.plot(fig, filename='sad/mnist/3ts_ano_gan_scores_cnn_comp')
