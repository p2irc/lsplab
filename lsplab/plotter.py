import matplotlib
matplotlib.use('Agg')

from . import stats

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os


new_dim = 2
alpha = 0.5
line_width = 0.3

def make_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def get_colormap(num_timepoints, condition):
    if condition == 'control':
        return [cm.Blues(x) for x in np.linspace(0.4, 1.0, num_timepoints)]
    else:
        return [cm.Reds(x) for x in np.linspace(0.4, 1.0, num_timepoints)]


def plot_general_ordination_plot(raw, output_path):
    num_timepoints = len(raw)

    colors_control = get_colormap(num_timepoints, 'control')
    colors_treated = get_colormap(num_timepoints, 'treated')
    treatments = [x[1] for x in raw[0]]
    features = np.array([item for sublist in [[x[2] for x in t] for t in raw] for item in sublist])

    make_directory(output_path)

    pca = stats.pca()
    pca.train(features, new_dim)
    trans = pca.transform(features)

    # Plotsky
    fig = plt.figure()

    treateds = []
    controls = []

    for i in range(0, len(trans), num_timepoints):
        t = treatments[i / num_timepoints]

        d = trans[i:i+num_timepoints]

        if t == 0:
            controls.append(d)
        else:
            treateds.append(d)

    treateds = np.array(treateds)
    controls = np.array(controls)

    plt.scatter(treateds[0][:, 0], treateds[0][:, 1], color=colors_treated, alpha=alpha)

    plt.scatter(controls[0][:, 0], controls[0][:, 1], color=colors_control, alpha=alpha)

    plt.savefig('{0}/general_ordination.png'.format(output_path))
    plt.close(fig)
