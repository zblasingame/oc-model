import numpy as np
import scipy.stats as ss
import csv

exploits = [
    'freak', 'poodle', 'nginx_keyleak', 'nginx_rootdir', 'logjam',
    'orzhttpd_rootdir', 'orzhttpd_restore'
]

models = ['aae', 'ano_gan_cnn', 'alad']

data = []

for model in models:
    for i, exploit in enumerate(exploits):
        ex_data = []
        for j in range(5):
            with open('../data/raid/{}/{}/subset_{}/test_set.csv'.format(
                model, exploit, j
            ), 'r') as f:
                entries = np.array([row for row in csv.reader(f)])
                # ex_data.append(np.column_stack((
                #     np.ones(entries.shape[0])*i,
                #     entries
                # )))
                ex_data.append(entries)

        ex_data = np.concatenate(ex_data, axis=0).astype(float)
        # scores = ex_data[:, 2]
        # labels = ex_data[:, 1]
        labels = ex_data[:, 0]
        scores = ex_data[:, 1]
        _min = np.min(scores[labels == 1])
        _max = np.max(scores[labels == 1])
        diff = _max - _min
        a = np.arctanh(-.5)
        b = np.arctanh(.5)
        alpha = (b-a)/diff
        beta = b - (alpha * _max)
        norm_scores = np.tanh(alpha * scores + beta)

        scores = norm_scores

        # ex_data[:, 2] = norm_scores

        # labels = ex_data[:, 0]
        # scores = ex_data[:, 1]

        dist = ss.wasserstein_distance(scores[labels == 1], scores[labels == 0])

        data.append(np.array([model, exploit, dist]).reshape(1, -1))

data = np.concatenate(data, axis=0).astype(str)
np.savetxt('../results/distances_norm.csv', data, delimiter=',', fmt='%s')


tab = """
\\begin{table}[h]
    \\centering
    \\begin{tabular}{lrrr}
    \\toprule
     & \\multicolumn{3}{c}{\\textbf{Models}}\\\\
    \\cmidrule{2-4}
    \\textbf{Exploit} & \\textbf{AAE} & \\textbf{AnoGAN} & \\textbf{ALAD}\\\\
    \\midrule
"""

av_data = []

for ex in exploits:
    nums = [float(data[np.logical_and(data[:, 1] == ex, data[:, 0] == m)][0, 2]) for m in models]
    tab += '    {} & {:.3f} & {:.3f} & {:.3f}\\\\\n'.format(ex.replace('_', '\\_'), *nums)
    av_data.append(np.array(nums).reshape(1, -1))

av_data = np.concatenate(av_data, axis=0)

av_data = np.mean(av_data, axis=0)

tab += '    {} & {:.3f} & {:.3f} & {:.3f}\\\\\n'.format('$\\mu$', *av_data)


tab += """
    \\bottomrule
    \\end{tabular}
    \\caption{Distance Table}
    \\label{tab:dist_table}
\\end{table}
"""

print(tab)
