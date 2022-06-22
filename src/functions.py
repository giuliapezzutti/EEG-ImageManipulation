import mne
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
        ax.legend()

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
        ax.legend()

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


def get_label(valence, arousal, threshold=0.1 * 4):

    if valence > 0 and arousal > 0:
        return 'HVHA'
    elif valence > 0 and arousal <= 0:
        return 'HVLA'
    elif valence <= 0 and arousal > 0:
        return 'LVHA'
    elif valence <= 0 and arousal <= 0:
        return 'LVLA'

def get_peak_pickle(epochs, channels_list, channels_interest, labels, t_min_epoch, fs, t_min, t_max, peak, mean=True):
    """
    Function to extract the peaks' amplitude from the epochs separately for each condition found and returns them
    or the mean value
    :param fs:
    :param epochs:
    :param channels_list:
    :param channels_interest: list of channels name to be investigated
    :param labels:
    :param t_min_epoch:
    :param t_min: lower bound of the time window in which the algorithm should look for the peak
    :param t_max: upper bound of the time window in which the algorithm should look for the peak
    :param peak: +1 for a positive peak, -1 for a negative peak
    :param mean: boolean value, if the return value should be the mean value or the list of amplitudes
    :return: if mean=True, mean amplitude value; otherwise list of detected peaks' amplitude and list of the
    correspondent annotations
    """

    peaks = {}
    annotations = {}

    # extraction of the data of interest and of the correspondent annotations
    channels_index = [i for i, e in enumerate(channels_list) if e in channels_interest]
    epochs_interest = epochs[:, channels_index, :]

    # get the unique conditions of interest
    if len(labels[0].split('/')) > 1:
        conditions_interest = [ann.split('/')[1] for ann in labels]
    else:
        conditions_interest = labels
    conditions_interest = list(set(conditions_interest))

    sample_min = int((t_min - t_min_epoch) * fs)
    sample_max = int((t_max - t_min_epoch) * fs)

    # for each condition of interest
    for condition in conditions_interest:

        # get the correspondent epochs and crop the signal in the time interval for the peak searching
        epochs_index = [i for i in range(epochs_interest.shape[0]) if condition in labels[i]]
        condition_roi_epoch = epochs_interest[epochs_index, :, sample_min:sample_max]
        condition_labels = np.array(labels)[epochs_index]

        # if necessary, get the annotation correspondent at each epoch
        # condition_labels = []
        # if not mean:
        #     condition_labels = [label for label in labels if '/' + condition in label]

        peak_condition, latency_condition, annotation_condition = [], [], []

        # for each epoch
        for idx, epoch in enumerate(condition_roi_epoch):

            # extract the mean signal between channels
            signal = np.array(epoch).mean(axis=0)

            # find location and amplitude of the peak of interest
            peak_loc, peak_mag = mne.preprocessing.peak_finder(signal, thresh=(max(signal) - min(signal)) / 50,
                                                               extrema=peak, verbose=False)
            peak_mag = peak_mag * 1e6

            # reject peaks too close to the beginning or to the end of the window
            if len(peak_loc) > 1 and peak_loc[0] == 0:
                peak_loc = peak_loc[1:]
                peak_mag = peak_mag[1:]
            if len(peak_loc) > 1 and peak_loc[-1] == (len(signal) - 1):
                peak_loc = peak_loc[:-1]
                peak_mag = peak_mag[:-1]

            # select peak according to the minimum or maximum one and convert the location from number of sample
            # (inside the window) to time instant inside the epoch
            if peak == -1:
                peak_loc = peak_loc[np.argmin(peak_mag)] / fs + t_min_epoch
                peak_mag = np.min(peak_mag)
            if peak == +1:
                peak_loc = peak_loc[np.argmax(peak_mag)] / fs + t_min_epoch
                peak_mag = np.max(peak_mag)

            # save the values found
            peak_condition.append(peak_mag)
            latency_condition.append(peak_loc)

            # in the not-mean case, it's necessary to save the correct labelling
            if not mean:
                annotation_condition.append(condition_labels[idx].split('/')[0])

        # compute output values or arrays for each condition
        if mean:
            peaks[condition] = np.mean(np.array(peak_condition))
        else:
            peaks[condition] = np.array(peak_condition)
            annotations[condition] = annotation_condition

    if not mean:
        return peaks, annotations

    return peaks
