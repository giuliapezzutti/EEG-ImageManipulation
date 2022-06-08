import glob
import json
import pickle

import numpy as np
import pandas as pd

if __name__ == '__main__':

    paths = glob.glob('../data/pickle/*.pkl')
    paths = [path.replace('\\', '/') for path in paths if '_data' in path]

    codes = [(path.rsplit('_', 1)[0]).rsplit('/', 1)[1] for path in paths]

    input_info = json.load(open('../data/eeg/info.json'))
    rois = input_info['rois']

    participant_data = pd.read_csv('../data/form-results/form-results.csv')
    participant_data = participant_data[['code', 'gender']]

    for idx, code in enumerate(codes):

        with open(paths[idx], 'rb') as f:
            data = pickle.load(f)
        with open(paths[idx].replace('_data', '_info'), 'rb') as f:
            info = pickle.load(f)
        with open(paths[idx].replace('_data', '_labels'), 'rb') as f:
            labels = pickle.load(f)
            conditions = list(set(labels))

        g = participant_data.loc[participant_data['code'] == code]['gender'].values[0]

        # calculate mean frontal amplitude in 300-600ms

        frontal_indexes = np.where(np.in1d(np.array(info['channels']), np.array(rois['frontal'])))[0]
        start = int((0.3 - info['tmin']) * info['fs'])
        end = int((0.6 - info['tmin']) * info['fs'])

        frontal_data = data[:, frontal_indexes, start:end]
        frontal_amplitude = np.mean(frontal_data, axis=-1)
        frontal_amplitude = np.mean(frontal_amplitude, axis=-1)

        pd_data = np.vstack((np.array(labels), frontal_amplitude)).T
        print(pd_data.shape)
        print(pd_data)
        exit(1)
