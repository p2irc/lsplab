import matplotlib
matplotlib.use('Agg')

from . import stats

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import os


new_dim = 2
alpha = 0.5
line_width = 0.3


def add_arrow(line):
    line = line[0]
    color = line.get_color()

    x_data = line.get_xdata()
    y_data = line.get_ydata()

    position = x_data.mean()

    start_ind = np.argmin(np.absolute(x_data - position))
    end_ind = start_ind - 1

    line.axes.annotate('',
                       xytext=(x_data[start_ind], y_data[start_ind]),
                       xy=(x_data[end_ind], y_data[end_ind]),
                       arrowprops=dict(arrowstyle="->", color=color),
                       size=12)


def lines_and_points(sample, colors, geno, i, snp, num_timepoints):
    x = sample[:, 0]
    y = sample[:, 1]

    plt.scatter(x, y, color=colors, alpha=alpha)

    if geno[i, snp] == 2.0:
        plt.plot(x, y, color='red', linewidth=line_width)
    else:
        plt.plot(x, y, color='black', linewidth=line_width)


def final_point(sample, colors, geno, i, snp, num_timepoints):
    x = sample[num_timepoints - 1, 0]
    y = sample[num_timepoints - 1, 1]

    if geno[i, snp] == 2.0:
        plt.scatter(x, y, color='red', alpha=alpha)
    else:
        plt.scatter(x, y, color='black', alpha=alpha)


def lines_only(sample, colors, geno, i, snp, num_timepoints):
    x = sample[:, 0]
    y = sample[:, 1]

    if geno[i, snp] == 2.0:
        line = plt.plot(x, y, color='red', alpha=alpha, linewidth=line_width)
    else:
        line = plt.plot(x, y, color='black', alpha=alpha, linewidth=line_width)


def single_lines(sample, colors, geno, i, snp, num_timepoints):
    x = [sample[0, 0], sample[num_timepoints - 1, 0]]
    y = [sample[0, 1], sample[num_timepoints - 1, 1]]

    if geno[i, snp] == 2.0:
        line = plt.plot(x, y, color='red', alpha=alpha, linewidth=line_width)
    else:
        line = plt.plot(x, y, color='black', alpha=alpha, linewidth=line_width)

    add_arrow(line)


def single_vectors(sample, colors, geno, i, snp, num_timepoints):
    x = [0., (sample[num_timepoints - 1, 0] - sample[0, 0])]
    y = [0., (sample[num_timepoints - 1, 1] - sample[0, 1])]

    if geno[i, snp] == 2.0:
        line = plt.plot(x, y, color='red', alpha=alpha, linewidth=line_width)
    else:
        line = plt.plot(x, y, color='black', alpha=alpha, linewidth=line_width)

    add_arrow(line)


def single_points(x, y):
    plt.scatter(x, y, color='blue', alpha=alpha)


def make_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def get_colormap(num_timepoints):
    return [cm.inferno(x) for x in np.linspace(0.0, 1.0, num_timepoints)]


def parfunc(num_samples, trans, colors, geno, snp, num_timepoints, output_path):
    fig = plt.figure()

    for i in range(num_samples):
        sample = trans[i, :, :]

        #lines_and_points(sample, colors, geno, i, snp, num_timepoints)
        #lines_only(sample, colors, geno, i, snp, num_timepoints)
        #single_lines(sample, colors, geno, i, snp, num_timepoints)
        single_vectors(sample, colors, geno, i, snp, num_timepoints)

    plt.savefig('{0}/{1}.png'.format(output_path, snp + 1))
    plt.close(fig)


def plot_ordination_from_datapoints_file(filename, output_path, num_snps, num_timepoints, original_dim, num_threads):
    num_timepoints = num_timepoints - 1
    colors = get_colormap(num_timepoints)

    make_directory(output_path)

    df = pd.read_csv(filename, sep='\t', header=None)
    raw = df.as_matrix()

    geno = raw[:, :num_snps]
    features = raw[:, num_snps:]

    # Transform features so they are one point per row
    features = np.reshape(features, [-1, original_dim])

    pca = stats.pca()
    pca.train(features, new_dim)
    trans = pca.transform(features)

    # Reshape so each row has multiple points
    trans = np.reshape(trans, [-1, num_timepoints, new_dim])
    num_samples = trans.shape[0]

    # Plotsky
    Parallel(n_jobs=num_threads)(delayed(parfunc)(num_samples, trans, colors, geno, snp, num_timepoints, output_path) for snp in tqdm(range(num_snps)))


def plot_general_ordination_plot(raw, output_path, num_timepoints, original_dim):
    num_timepoints = num_timepoints - 1
    colors = get_colormap(num_timepoints)

    make_directory(output_path)

    # Transform features so they are one point per row
    features = np.reshape(raw, [-1, original_dim])

    pca = stats.pca()
    pca.train(features, new_dim)
    trans = pca.transform(features)

    # Reshape so each row has multiple points
    #trans = np.reshape(trans, [-1, num_timepoints, new_dim])
    #num_samples = trans.shape[0]

    # Plotsky
    fig = plt.figure()

    # samples_X = []
    # samples_Y = []
    #
    # for sample in trans:
    #     samples_X.append(sample[num_timepoints - 1, 0] - sample[0, 0])
    #     samples_Y.append(sample[num_timepoints - 1, 1] - sample[0, 1])

    single_points(trans[:, 0], trans[:, 1])

    plt.savefig('{0}/general_ordination.png'.format(output_path))
    plt.close(fig)


def plot_path(output_path, name, features):
    make_directory(output_path)

    pca = stats.pca()
    pca.train(features, new_dim)
    trans = pca.transform(features)

    # Plotsky
    fig = plt.figure()

    samples_X = []
    samples_Y = []

    for i in range(len(trans)):
        sample = trans[i, :]
        samples_X.append(sample[0])
        samples_Y.append(sample[1])

    single_points(samples_X, samples_Y)

    plt.savefig('{0}/{1}.png'.format(output_path, name))
    plt.close(fig)
    #plt.show()


def plot_embeddings_from_projections_file(filename, meta_filename, output_path, num_timepoints, original_dim):
    num_timepoints = num_timepoints - 1
    colors = get_colormap(num_timepoints)

    make_directory(output_path)

    df = pd.read_csv(filename, sep='\t', header=None)
    features = df.as_matrix()

    meta_df = pd.read_csv(meta_filename, header=None)
    meta = meta_df.as_matrix()

    pca = stats.pca()
    pca.train(features, new_dim)
    trans = pca.transform(features)

    # Reshape so each row hs multiple points
    num_samples = trans.shape[0]

    # Plotsky
    fig = plt.figure()

    for i in range(num_samples):
        sample = trans[i, :]
        label = meta[i]
        x = sample[0]
        y = sample[1]

        plt.scatter(x, y, color=colors[int(label[0])], alpha=0.5)

    plt.savefig('{0}/{1}.png'.format(output_path, os.path.basename(filename)))
    plt.close(fig)
