import json
import tkinter

from EEGAnalysis import *
from functions import *

if __name__ == '__main__':

    paths = ['../data/eeg/subj_viwi30_block1.xdf', '../data/eeg/subj_ervi22_block1.xdf',
             '../data/eeg/subj_vamo24_block1.xdf', '../data/eeg/subj_mama13_block1.xdf',
             '../data/eeg/subj_moob25_block1.xdf', '../data/eeg/subj_mile27_block1.xdf',
             '../data/eeg/subj_jomo20_block1.xdf', '../data/eeg/subj_vasa28_block1.xdf']

    # ['../data/eeg/subj_maba09_block1.xdf', '../data/eeg/subj_soze31_block1.xdf',
    # '../data/eeg/subj_nipe10_block1.xdf', '../data/eeg/subj_dino02_block1.xdf']
    # ['../data/eeg/subj_ervi22_block1.xdf', '../data/eeg/subj_viwi30_block1.xdf',
    # '../data/eeg/subj_vamo24_block1.xdf', '../data/eeg/subj_mama13_block1.xdf',
    # '../data/eeg/subj_moob25_block1.xdf', '../data/eeg/subj_mile27_block1.xdf',
    # '../data/eeg/subj_jomo20_block1.xdf', '../data/eeg/subj_vasa28_block1.xdf']

    dict_info = json.load(open('../data/eeg/info.json'))

    signals_means = {}

    for path in paths:

        plt.close('all')
        print('\n\nAnalyzing file', path)

        eeg = EEGAnalysis(path, dict_info)
        eeg.run_raw_epochs(visualize_raw=False, save_images=True)

        if len(paths) > 1:
            evoked = eeg.evoked
            for key in evoked.keys():
                if key in signals_means:
                    signals_means[key] = mne.combine_evoked([signals_means[key], evoked[key]], weights='equal')
                else:
                    signals_means[key] = evoked[key]

    conditions, rois = derive_conditions_rois(labels=signals_means.keys())
    plot_mean_epochs(signals_means, conditions, rois, dict_info['erp'])
