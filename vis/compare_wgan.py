import numpy as np
import csv

import plotly.plotly as py
import plotly.graph_objs as go


with open('../results/raid-net-knn.csv', 'r') as f:
    data = np.array([row for row in csv.reader(f)])

with open('../results/raid-net-knn-wgan.csv') as f:
    w_data = np.array([row for row in csv.reader(f)])

exploits = np.unique(data[:, 0])

traces = [go.Scatter(
    x=exploits,
    y=[np.mean(data[data[:, 0] == ex, 1].astype(np.float)) for ex in exploits],
    mode='lines',
    name='KNN avg'
), go.Scatter(
    x=exploits,
    y=[np.mean(w_data[w_data[:, 0] == ex, 1].astype(np.float)) for ex in exploits],
    mode='lines',
    name='W-Classifier avg'
), go.Scatter(
    x=exploits,
    y=[np.max(data[data[:, 0] == ex, 1].astype(np.float)) for ex in exploits],
    mode='lines',
    name='KNN max'
), go.Scatter(
    x=exploits,
    y=[np.max(w_data[w_data[:, 0] == ex, 1].astype(np.float)) for ex in exploits],
    mode='lines',
    name='W-Classifier max'
)
]

layout = go.Layout(
    title='W-Classifier vs KNN+Softmax',
    yaxis=dict(title='Accuracy (%)')
)

fig = go.Figure(data=traces, layout=layout)
py.plot(fig, filename='sad/ursad-raid-3ts-knn-vs-wclassifier')
