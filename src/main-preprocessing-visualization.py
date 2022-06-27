import json
from EEGAnalysis import EEGAnalysis
from functions import *

if __name__ == '__main__':

    paths = ['../data/eeg/subj_mama13_block1.xdf']

    # ['../data/eeg/subj_maba09_block1.xdf', '../data/eeg/subj_soze31_block1.xdf',
    # '../data/eeg/subj_nipe10_block1.xdf', '../data/eeg/subj_dino02_block1.xdf']
    # ['../data/eeg/subj_brfr09_block1.xdf', '../data/eeg/subj_riri06_block1.xdf',
    # '../data/eeg/subj_keaz01_block1.xdf', '../data/eeg/subj_jemc16_block1.xdf',
    # '../data/eeg/subj_viwi30_block1.xdf', '../data/eeg/subj_ervi22_block1.xdf',
    # '../data/eeg/subj_vamo24_block1.xdf', '../data/eeg/subj_mama13_block1.xdf',
    # '../data/eeg/subj_moob25_block1.xdf', '../data/eeg/subj_mile27_block1.xdf',
    # '../data/eeg/subj_jomo20_block1.xdf', '../data/eeg/subj_vasa28_block1.xdf']

    dict_info = json.load(open('../data/eeg/info.json'))
    dict_info_full = json.load(open('../data/eeg/info_full.json'))

    signals_means = {}

    for path in paths:

        plt.close('all')
        print('\n\nAnalyzing file', path)

        # eeg = EEGAnalysis(path, dict_info)
        # eeg.run_raw_epochs(visualize_raw=False, save_images=True, ica_analysis=False, create_evoked=True,
        #                    save_pickle=False)
        #
        # if len(paths) > 1:
        #     evoked = eeg.evoked
        #     for key in evoked.keys():
        #         if key in signals_means:
        #             signals_means[key] = mne.combine_evoked([signals_means[key], evoked[key]], weights='equal')
        #         else:
        #             signals_means[key] = evoked[key]

        eeg = EEGAnalysis(path, dict_info_full)
        eeg.run_raw_epochs(visualize_raw=False, save_images=False, create_evoked=False, save_pickle=True)

        exit(1)

    conditions, rois = derive_conditions_rois(labels=signals_means.keys())
    plot_mean_epochs(signals_means, conditions, rois, dict_info['erp'])
