import json

from src.EEGAnalysis import EEGAnalysis

if __name__ == '__main__':

    path = 'data/eeg/subj_ervi22_block1.xdf'
    dict_info = json.load(open('../data/eeg/info.json'))

    eeg = EEGAnalysis(path, dict_info)
    eeg.run_whole(visualize_raw=False, save_images=True)
