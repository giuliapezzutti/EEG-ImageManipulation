import numpy as np
from matplotlib import pyplot as plt
import pylab as py
from scipy import optimize


def create_personality_matrix(num_personalities, num_data, personality_types):
    """
    Creation of multiplication matrix and bias vector for the computation of the personality test according to the
    definition
    :param personality_types:
    :param num_personalities: number of personalities types in the study
    :param num_data: number of data to which the subject has answered
    :return: multiplication matrix and bias vector
    """

    # empty personality matrix
    personality_matrix = np.zeros([num_personalities, num_data])

    # where to put +1 or -1 in the personality matrix for each row
    E = {'name': 'E', '+': [1, 11, 21, 31, 41], '-': [6, 16, 26, 36, 46]}
    A = {'name': 'A', '+': [7, 17, 27, 37, 42, 47], '-': [2, 12, 22, 32]}
    C = {'name': 'C', '+': [3, 13, 23, 33, 43, 48], '-': [8, 18, 28, 38]}
    N = {'name': 'N', '+': [9, 19], '-': [4, 14, 24, 29, 34, 39, 44, 49]}
    O = {'name': 'O', '+': [5, 15, 25, 35, 40, 45, 50], '-': [10, 20, 30]}

    # filling of the matrix according to the definition
    for dict in [E, A, C, N, O]:

        name = dict['name']
        plus = dict['+']
        minus = dict['-']

        index = personality_types.index(name)

        for idx in plus:
            personality_matrix[index, idx - 1] = +1
        for idx in minus:
            personality_matrix[index, idx - 1] = -1

    # personality bias vector definition according to the explanation
    personality_bias = [20, 14, 14, 38, 8]

    return personality_matrix, personality_bias


def derive_conditions_rois(labels):
    conditions = [s.split('/')[0] for s in labels]
    conditions = list(set(conditions))
    rois = [s.split('/')[1] for s in labels]
    rois = list(set(rois))
    return conditions, rois


def plot_mean_epochs(mean_signals, conditions, rois, erps):
    conditions = sorted(conditions)
    rois = sorted(rois)

    x_axis = mean_signals['blackwhite/central'].times * 1000

    fig, axs = plt.subplots(3, 2, figsize=(25.6, 19.2))

    path = '../images/epochs/manipulations.png'

    min_value = np.inf
    max_value = -np.inf

    for _, evoked in mean_signals.items():
        data = evoked.get_data()[0]
        min_value = min(min_value, min(data))
        max_value = max(max_value, max(data))

    for i, ax in enumerate(fig.axes):

        condition = conditions[i]
        correct_labels = [s for s in mean_signals.keys() if condition + '/' in s]
        correct_short_labels = [s.split('/')[1] for s in correct_labels]

        for idx, label in enumerate(correct_labels):
            ax.plot(x_axis, mean_signals[label].get_data()[0], label=correct_short_labels[idx])

        for erp in erps:
            ax.vlines(erp, ymin=min_value, ymax=max_value, linestyles='dashed')

        ax.set_xlabel('Time (\u03bcs)')
        ax.set_ylabel('Amplitude (V)')
        ax.set_title(condition)

    plt.legend(bbox_to_anchor=(1.2, 2))
    plt.savefig(path)
    plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(25.6, 19.2))
    path = '../images/epochs/rois.png'

    for i, ax in enumerate(fig.axes):

        roi = rois[i]

        correct_labels = [s for s in mean_signals.keys() if '/' + roi in s]
        correct_short_labels = [s.split('/')[0] for s in correct_labels]

        for idx, label in enumerate(correct_labels):
            ax.plot(x_axis, mean_signals[label].get_data()[0], label=correct_short_labels[idx])

        for erp in erps:
            ax.vlines(erp, ymin=min_value, ymax=max_value, linestyles='dashed')

        ax.set_xlabel('Time (\u03bcs)')
        ax.set_ylabel('Amplitude (V)')
        ax.set_title(roi)

    plt.legend(bbox_to_anchor=(1.2, 1.1))
    plt.savefig(path)
    plt.close()


def get_fitted_normal_distribution(data, number_bins=100):
    # Equation for Gaussian
    def f(x, a, b, c):
        return a * py.exp(-(x - b) ** 2.0 / (2 * c ** 2))

    # Generate data from bins as a set of points
    x = [0.5 * (data[1][i] + data[1][i + 1]) for i in range(len(data[1]) - 1)]
    y = data[0]

    popt, pcov = optimize.curve_fit(f, x, y)

    x_fit = py.linspace(x[0], x[-1], number_bins)
    y_fit = f(x_fit, *popt)

    return x_fit, y_fit


def plot_distribution(array_data, path):
    bins = np.linspace(array_data.min(), array_data.max(), 100)
    data = py.hist(array_data, bins=bins)

    x_fit, y_fit = get_fitted_normal_distribution(data, number_bins=len(bins))
    plt.plot(x_fit, y_fit, lw=4, color="r")

    plt.title((path.rsplit('.', 1)[0]).rsplit('/', 1)[1])
    plt.savefig(path)
    plt.close()

