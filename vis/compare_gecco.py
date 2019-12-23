import numpy as np
import csv

import plotly.plotly as py
import plotly.graph_objs as go


with open('../results/tc-3ts-gp-hellinger.csv', 'r') as f:
    data = np.array([row for row in csv.reader(f)])

with open('../results/tc-3ts-gp-paper-hellinger.csv') as f:
    p_data = np.array([row for row in csv.reader(f)])

with open('../results/tc-3ts-gp-hellinger-discrete.csv') as f:
    d_data = np.array([row for row in csv.reader(f)])

with open('../results/tc-3ts-gp-hellinger-unnorm.csv') as f:
    u_data = np.array([row for row in csv.reader(f)])

with open('../results/tc-3ts-gp-hellinger-full.csv') as f:
    f_data = np.array([row for row in csv.reader(f)])


# exploits = np.unique(data[:, 0])
# exploits = data[]

# traces = [go.Scatter(
#     x=exploits,
#     y=[np.mean(data[data[:, 0] == ex, 1].astype(np.float)) for ex in exploits],
#     marker=dict(color='#1abc9c'),
#     name='KDE P(f(X|Y=i)) avg'
# ), go.Scatter(
#     x=exploits,
#     y=[np.mean(p_data[p_data[:, 0] == ex, 1].astype(np.float)) for ex in exploits],
#     marker=dict(color='#3498db'),
#     name='KDE P(Y=i|f(X)) avg'
# ), go.Scatter(
#     x=exploits,
#     y=[np.mean(p_data[p_data[:, 0] == ex, 1].astype(np.float)) for ex in exploits],
#     marker=dict(color='#34495e'),
#     name='KDE P(f(X|Y=i)) avg'
# ), go.Scatter(
#     x=exploits,
#     y=[np.max(data[data[:, 0] == ex, 1].astype(np.float)) for ex in exploits],
#     marker=dict(color='#1abc9c'),
#     name='KDE P(f(X|Y=i)) max'
# ), go.Scatter(
#     x=exploits,
#     y=[np.max(p_data[p_data[:, 0] == ex, 1].astype(np.float)) for ex in exploits],
#     marker=dict(color='#3498db'),
#     name='KDE P(Y=i|f(X)) max'
# ), go.Scatter(
#     x=exploits,
#     y=[np.mean(d_data[d_data[:, 0] == ex, 1].astype(np.float)) for ex in exploits],
#     marker=dict(color='#e74c3c'),
#     name='Discrete P(f(X|Y=i)) avg'
# ), go.Scatter(
#     x=exploits,
#     y=[np.max(d_data[d_data[:, 0] == ex, 1].astype(np.float)) for ex in exploits],
#     marker=dict(color='#e74c3c'),
#     name='Discrete P(f(X|Y=i)) max'
# )
# ]


traces = [go.Box(
    x=data[:, 0],
    y=data[:, 1],
    name='KDE P(f(X|Y=i))',
    marker=dict(color='#1abc9c')
), go.Box(
    x=data[:, 0],
    y=p_data[:, 1],
    name='KDE P(Y=i|f(X))',
    marker=dict(color='#3498db')
), go.Box(
    x=data[:, 0],
    y=d_data[:, 1],
    name='Discrete P(f(X|Y=i))',
    marker=dict(color='#e74c3c')
), go.Box(
    x=data[:, 0],
    y=f_data[:, 1],
    name='KDE Full Hellinger P(f(X|Y=i))',
    marker=dict(color='#9b59b6')
), go.Box(
    x=data[:, 0],
    y=u_data[:, 1],
    name='KDE Unnormalized GP P(f(X|Y=i))',
    marker=dict(color='#34495e')
)]

layout = go.Layout(
    title='Hellinger Distance Implementations',
    boxmode='group',
    yaxis=dict(title='Accuracy (%)')
)

fig = go.Figure(data=traces, layout=layout)
py.plot(fig, filename='gecco/hellinger_version_comparison')
