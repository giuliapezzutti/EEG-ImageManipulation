import json
from os.path import exists

import pandas as pd
from sklearn import preprocessing

from EEGAnalysis import *

if __name__ == '__main__':

    # get the codes of people in the form
    path_form = '../data/form-results/form-results.csv'
    df_form = pd.read_csv(path_form, index_col=0)
    codes_form = df_form.loc[:, 'code'].tolist()
    codes_form = list(set(codes_form))

    # get the codes of people in the ratings
    path_ratings = '../data/ratings-results/ratings-results.csv'
    df_ratings = pd.read_csv(path_ratings, index_col=0)
    codes_ratings = df_ratings.loc[:, 'code'].tolist()
    codes_ratings = list(set(codes_ratings))

    path_eeg = '../data/eeg/'
    dict_info = json.load(open('../data/eeg/info_full.json'))

    info_dataset, signal_dataset, label_dataset = [], [], []

    for code in codes_form:

        if code not in codes_ratings:
            continue

        if code == 'krki20' or code == 'nipe10' or code == 'maba09' or code == 'soze31' or code == 'dino02':
            continue

        print(code)

        with open('../data/pickle/'+code+'_data.pkl', 'rb') as f:
            data = pickle.load(f)
        with open('../data/pickle/'+code+'_info.pkl', 'rb') as f:
            info = pickle.load(f)
            index_channels_eeg = [index for index in range(len(info['channels'])) if
                                  'EOG' not in info['channels'][index]]
            channels_eeg = np.array(info['channels'])[index_channels_eeg]
        with open('../data/pickle/'+code+'_labels.pkl', 'rb') as f:
            labels = pickle.load(f)
            conditions = [label.split('/')[1] for label in labels]
            unique_conditions = list(set(conditions))

        data_form = df_form.loc[df_form['code'] == code, :].values.flatten().tolist()[1:]
        form = np.array([data_form] * len(data))

        data_ratings = df_ratings.loc[df_ratings['code'] == code]
        ratings = []

        for idx, _ in enumerate(data):

            img_name = labels[idx].split('/')[0]
            ratings.append(data_ratings.loc[data_ratings['img_name'] == img_name][['valence', 'arousal']].values[0])

        ratings = np.array(ratings)

        signal_dataset.append(data)
        info_dataset.append(form)
        label_dataset.append(ratings)

    info_dataset = np.ndarray(info_dataset)
    signal_dataset = np.ndarray(signal_dataset)
    label_dataset = np.ndarray(label_dataset)

    print(info_dataset)
    print(signal_dataset.shape)
    print(label_dataset.shape)

    # for i in [0, 2, 3]:
    #     encoder = preprocessing.LabelEncoder()
    #     info_dataset[:, i] = encoder.fit_transform(info_dataset[:, i])
    #
    # for i in range(4, 9):
    #     info_dataset[:, i] = np.array(info_dataset[:, i], dtype=float)/40
    #
    # threshold = 0.1*4
    # label_binary_dataset = []
    # for labels in label_dataset:
    #     valence = labels[0]
    #     arousal = labels[1]
    #
    #     if (np.square(valence) + np.square(arousal)) <= np.square(threshold):
    #         label_binary_dataset.append([0, 0, 1])
    #     elif valence > 0 and arousal > 0:
    #         label_binary_dataset.append([1, 1, 0])
    #     elif valence > 0 and arousal <= 0:
    #         label_binary_dataset.append([1, 0, 0])
    #     elif valence <= 0 and arousal > 0:
    #         label_binary_dataset.append([0, 1, 0])
    #     elif valence <= 0 and arousal <= 0:
    #         label_binary_dataset.append([0, 0, 0])
    #
    # Path('../data/final-dataset/').mkdir(parents=True, exist_ok=True)
    #
    # np.save('../data/final-dataset/info_dataset.npy', info_dataset)
    # np.save('../data/final-dataset/signal_dataset.npy', signal_dataset)
    # np.save('../data/final-dataset/label_dataset.npy', label_binary_dataset)
