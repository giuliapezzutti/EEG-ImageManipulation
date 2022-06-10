import glob
import json
import pickle
import re

import mne
import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

if __name__ == '__main__':

    paths = glob.glob('../data/pickle/*.pkl')
    paths = [path.replace('\\', '/') for path in paths if '_data' in path]

    codes = [(path.rsplit('_', 1)[0]).rsplit('/', 1)[1] for path in paths]

    input_info = json.load(open('../data/eeg/info.json'))
    rois = input_info['rois']

    participant_data = pd.read_csv('../data/form-results/form-results.csv')
    participant_data = participant_data[['code', 'gender']]

    ratings_data = pd.read_csv('../data/ratings-results/ratings-results.csv')
    ratings_data = ratings_data[['code', 'image_name', 'valence', 'arousal']]

    pd_signal = None
    pd_gender = []

    for idx, code in enumerate(codes):

        print(code)

        with open(paths[idx], 'rb') as f:
            data = pickle.load(f)
        with open(paths[idx].replace('_data', '_info'), 'rb') as f:
            info = pickle.load(f)
        with open(paths[idx].replace('_data', '_labels'), 'rb') as f:
            labels = pickle.load(f)
            conditions = [label.split('/')[1] for label in labels]
            unique_conditions = list(set(conditions))

        data = data[:, [index for index in range(len(info['channels'])) if 'EOG' not in info['channels'][index]]]

        g = participant_data.loc[participant_data['code'] == code]['gender'].values[0]

        epochs = data.reshape(data.shape[0], -1)
        gender = np.expand_dims(np.array([g]*(epochs.shape[0])), axis=-1)

        if pd_signal is None:
            pd_signal = epochs
            pd_gender = gender
        else:
            pd_signal = np.concatenate((pd_signal, epochs), axis=0)
            pd_gender = np.concatenate((pd_gender, gender), axis=0)

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

    # pd = pd.DataFrame(data=pd_data, columns=['manipulation', 'f-amp', 'tl-peak'])
    # print(pd)

    print('\n\nGender prediction from whole signals')
    pd_signal = np.array(pd_signal)

    X = pd.DataFrame(pd_signal)
    Y = pd.DataFrame(data=pd_gender, columns=['gender'])

    df = pd.concat([X, Y], axis=1)

    Y = Y[['gender']] == 'female'
    Y = np.ravel(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

    model = LDA()
    model.fit(X_train, y_train)

    print('Train score: ', model.score(X_train, y_train))
    print('Test score: ', model.score(X_test, y_test))

    coefs = np.array(model.coef_)
    coefs = coefs.reshape(-1, data.shape[-1])

    seaborn.heatmap(coefs)
    plt.show()
