import json
from os.path import exists

import pandas as pd
from sklearn import preprocessing

from EEGAnalysis import *

if __name__ == '__main__':

    # get the codes of people in the form
    path_form = '../data/form-results/form-results.csv'
    df_form = pd.read_csv(path_form, index_col=0)
    codes_form = df_form.loc[:, 'participant code'].tolist()
    codes_form = list(set(codes_form))

    # get the codes of people in the ratings
    path_ratings = '../data/ratings-results/ratings-results.csv'
    df_ratings = pd.read_csv(path_ratings, index_col=0)
    codes_ratings = df_ratings.loc[:, 'participant code'].tolist()
    codes_ratings = list(set(codes_ratings))

    path_eeg = '../data/eeg/'
    dict_info = json.load(open('../data/eeg/info_full.json'))

    info_dataset, signal_dataset, label_dataset = [], [], []

    for code in codes_form:

        if code not in codes_ratings:
            continue

        path_signals = path_eeg+'/subj_'+code+'_block1.xdf'
        if not exists(path_signals):
            continue

        print('\n\n\ncode')

        data_form = df_form.loc[df_form['participant code'] == code, :].values.flatten().tolist()[1:]
        data_ratings = df_ratings.loc[df_ratings['participant code'] == code]

        eeg = EEGAnalysis(path_signals)
        eeg.create_raw()
        eeg.raw_time_filtering()
        eeg.raw_spatial_filtering()
        eeg.create_epochs()

        signals = eeg.get_epochs_dataframe()
        signals.to_csv('../data/eeg/signal.csv')

        epochs = signals.loc[:, 'epoch'].values.flatten()
        epochs = np.unique(epochs)

        for epoch in epochs:

            current_signal = signals.loc[signals['epoch'] == epoch]
            x = current_signal.iloc[:, 3:].to_numpy().T

            img_name = current_signal.iloc[0]['condition']
            img_name = img_name.split('/')[0]

            ratings = data_ratings.loc[data_ratings['img_name'] == img_name][['valence', 'arousal']].values[0]

            info_dataset.append(data_form)
            signal_dataset.append(x)
            label_dataset.append(ratings)

    info_dataset = np.array(info_dataset)
    signal_dataset = np.array(signal_dataset)
    label_dataset = np.array(label_dataset)

    for i in [0, 2, 3]:
        encoder = preprocessing.LabelEncoder()
        info_dataset[:, i] = encoder.fit_transform(info_dataset[:, i])

    for i in range(4, 9):
        info_dataset[:, i] = np.array(info_dataset[:, i], dtype=float)/40

    threshold = 0.1*4
    label_binary_dataset = []
    for labels in label_dataset:
        valence = labels[0]
        arousal = labels[1]

        if (np.square(valence) + np.square(arousal)) <= np.square(threshold):
            label_binary_dataset.append([0, 0, 1])
        elif valence > 0 and arousal > 0:
            label_binary_dataset.append([1, 1, 0])
        elif valence > 0 and arousal <= 0:
            label_binary_dataset.append([1, 0, 0])
        elif valence <= 0 and arousal > 0:
            label_binary_dataset.append([0, 1, 0])
        elif valence <= 0 and arousal <= 0:
            label_binary_dataset.append([0, 0, 0])

    Path('../data/final-dataset/').mkdir(parents=True, exist_ok=True)

    np.save('../data/final-dataset/info_dataset.npy', info_dataset)
    np.save('../data/final-dataset/signal_dataset.npy', signal_dataset)
    np.save('../data/final-dataset/label_dataset.npy', label_binary_dataset)
