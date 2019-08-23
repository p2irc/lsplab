import matplotlib
matplotlib.use('Agg')

from . import stats

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


new_dim = 2
alpha = 0.5
line_width = 0.3


def get_colormap(num_timepoints, condition):
    if condition == 'control':
        return [cm.Blues(x) for x in np.linspace(0.4, 1.0, num_timepoints)]
    else:
        return [cm.Reds(x) for x in np.linspace(0.4, 1.0, num_timepoints)]


def plot_general_ordination_plot(raw, output_path):
    num_timepoints = len(raw)

    colors_control = get_colormap(num_timepoints, 'control')
    colors_treated = get_colormap(num_timepoints, 'treated')
    features = np.array([item for sublist in [[x[2] for x in t] for t in raw] for item in sublist])

    treatments = [row[1] for row in raw[0]] * num_timepoints

    pca = stats.pca()
    pca.train(features, new_dim)
    trans = pca.transform(features)

    trans = trans.reshape([num_timepoints, -1, new_dim])

    # Plotsky
    fig = plt.figure()

    for i in range(len(trans[0])):
        t = treatments[i]

        d = np.array([x[i] for x in trans])

        if t == 1:
            plt.scatter(d[:, 0], d[:, 1], color=colors_treated, alpha=alpha)
        else:
            plt.scatter(d[:, 0], d[:, 1], color=colors_control, alpha=alpha)

    plt.savefig(output_path)
    plt.close(fig)
