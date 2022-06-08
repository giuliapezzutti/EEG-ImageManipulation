import os
import sys
import tkinter

from matplotlib import pyplot as plt

from ..EEGAnalysis import *


def get_path():

    if 'tkinter' in sys.modules:
        from tkinter import filedialog
        path_selected = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select a File",
                                                   filetypes=(("xdf files", "*.xdf*"),))
    else:
        path_selected = input(
            "Not able to use tkinter to select the file. Insert here the file path and press ENTER:\n")

    return path_selected


def get_subject_name(filepath):
    base = os.path.basename(filepath)
    file = os.path.splitext(base)[0]

    z = file.split('-', 1)[1]
    z = z.split('_ses', 1)[0]

    return z


if __name__ == '__main__':

    path = get_path()
    # path = '../data/sub-test_without_gel/ses-S001/eeg/sub-test_without_gel_ses-S001_task-Default_run-001_eeg.xdf'
    filename = get_subject_name(path)
    eeg = EEGAnalysis(path, dict_info=None)  # to be fixed if it must be executed
    eeg.create_raw()
    eeg.raw_time_filtering()
    eeg.visualize_raw(psd_topo=False)
    eeg_data = eeg.get_raw_ndarray().T

    eeg_fs = eeg.__get_eeg_fs__()

    samples_to_be_removed = eeg_fs * 60
    eeg_data = eeg_data[samples_to_be_removed:samples_to_be_removed+200*eeg_fs, :]

    fig, axs = plt.subplots(eeg_data.shape[1], 4, figsize=(25, 16), gridspec_kw={'width_ratios': [3, 3, 3, 3]})
    fig.subplots_adjust(wspace=0.5)

    for channel in range(eeg_data.shape[1]):
        eeg_current = np.array(eeg_data[:, channel]).flatten()

        eeg_filt = np.array(eeg_data[:, channel]).flatten()

        x = np.arange(len(eeg_filt)) / eeg_fs
        axs[channel, 0].set_xlabel('Time (s)', fontsize=10)
        axs[channel, 0].set_ylabel('Channel ' + str(channel + 1) + '\n\n\nAmplitude (uV)'.format(channel), fontsize=10)
        axs[channel, 0].yaxis.set_label_coords(-0.2, 0.5)
        # axs[channel, 0].set_ylim([-10, 10])
        # axs[channel, 0].set_xlim([0, 200])
        axs[channel, 0].plot(x, eeg_filt)

        x = np.arange(500) / eeg_fs
        axs[channel, 1].set_xlabel('Time (s)', fontsize=10)
        axs[channel, 1].set_ylabel('Amplitude (uV)', fontsize=10)
        axs[channel, 1].yaxis.set_label_coords(-0.2, 0.5)
        axs[channel, 1].plot(x, eeg_filt[0:500])

        bins = np.linspace(-6, 6, 100)
        axs[channel, 2].set_xlabel('Amplitude', fontsize=10)
        axs[channel, 2].set_ylabel('Frequency', fontsize=10)
        axs[channel, 2].yaxis.set_label_coords(-0.2, 0.5)
        # axs[channel, 2].set_ylim([0, 5000])
        axs[channel, 2].hist(eeg_filt, bins, alpha=0.5, histtype='bar', ec='black')

        axs[channel, 3].magnitude_spectrum(eeg_filt, Fs=eeg_fs)
        axs[channel, 3].set_xlabel('Frequency (Hz)', fontsize=10)
        axs[channel, 3].set_ylabel('Amplitude (uV)', fontsize=10)
        axs[channel, 3].yaxis.set_label_coords(-0.2, 0.5)
        # axs[channel, 3].set_ylim([0, 0.2])
        axs[channel, 3].set_xlim(-2, 42)

    fig.suptitle(filename, fontsize=30)
    plt.savefig('../images/{}.jpg'.format(filename))
    fig.show()
