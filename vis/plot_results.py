import numpy as np
import csv

import plotly.plotly as py
import plotly.graph_objs as go


with open('../results/raid-net-int-new.csv', 'r') as f:
    data = np.array([row for row in csv.reader(f)])

exploits = np.unique(data[:, 0])

boxes = [go.Box(
    y=data[data[:, 0] == ex, 1].astype(np.float),
    name=ex
) for ex in exploits]

layout = go.Layout(
    title='Performance of Softmax Net with Nearest Neighbour scoring',
    yaxis=dict(title='Accuracy (%)')
)

fig = go.Figure(data=boxes, layout=layout)
py.plot(fig, filename='sad/ursad-raid-net-nn-scoring-results')
