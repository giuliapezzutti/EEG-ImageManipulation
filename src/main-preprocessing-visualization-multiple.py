import json
from EEGAnalysis import *
from functions import *

if __name__ == '__main__':

    paths = ['../data/eeg/subj_thko03_block1.xdf', '../data/eeg/subj_thko03_block3.xdf',
             '../data/eeg/subj_thko03_block2.xdf']

    dict_info = json.load(open('../data/eeg/info.json'))
    signals_means = {}

    raws = []
    annotations_dict = {}

    for path in paths[1:]:

        plt.close('all')
        print('\n\nAnalyzing file', path)

        eeg = EEGAnalysis(path, dict_info)
        eeg.run_raw(visualize_raw=False)

        raw = eeg.raw
        raws.append(raw)

    eeg = EEGAnalysis(paths[0], dict_info)
    eeg.run_combine_raw_epochs(visualize_raw=False, save_images=False, new_raws=raws)
