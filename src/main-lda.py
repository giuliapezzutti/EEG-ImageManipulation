import glob
import json
import pickle
import re

import mne
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

        epochs = data.reshape(data.shape[0], -1)
        gender = np.expand_dims(np.array([g]*(epochs.shape[0])), axis=-1)
        pd_data = np.concatenate([gender, epochs], axis=1)

        # calculate mean frontal amplitude in 300-600ms

        frontal_indexes = np.where(np.in1d(np.array(info['channels']), np.array(rois['frontal'])))[0]
        start = int((0.3 - info['tmin']) * info['fs'])
        end = int((0.6 - info['tmin']) * info['fs'])

        frontal_data = data[:, frontal_indexes, start:end]
        frontal_amplitude = np.mean(frontal_data, axis=-1)
        frontal_amplitude = np.mean(frontal_amplitude, axis=-1)*1e6

        # calculate temporal left N200

        temporal_channels = [channel for channel in rois['temporal'] if int(re.sub('\D', '', channel)) % 2 == 1]
        temporal_indexes = np.where(np.in1d(np.array(info['channels']), np.array(temporal_channels)))[0]

        start = int((0.17 - info['tmin']) * info['fs'])
        end = int((0.23 - info['tmin']) * info['fs'])

        temporal_data = data[:, temporal_indexes, start:end]
        temporal_data = np.mean(temporal_data, axis=1)

        peaks = []
        for epoch in temporal_data:
            peak_loc, peak_mag = mne.preprocessing.peak_finder(epoch, thresh=(max(epoch) - min(epoch)) / 50,
                                                               extrema=-1, verbose=False)
            peaks.append(np.min(peak_mag)*1e6)

        pd_data = np.vstack((np.array(labels), frontal_amplitude, np.array(peaks))).T

        pd = pd.DataFrame(data=pd_data, columns=['manipulation', 'f-amp', 'tl-peak'])
        print(pd)

        exit(1)
