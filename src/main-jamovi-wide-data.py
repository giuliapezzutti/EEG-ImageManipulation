import json
import pickle

import numpy as np
import pandas as pd

from src.EEGAnalysis import EEGAnalysis
from src.functions import get_peak_pickle

if __name__ == '__main__':

    # read form-result file to get codes and gender of the participants
    participant_data = pd.read_csv('../data/form-results/form-results.csv')
    participant_data = participant_data[['code', 'gender']]

    # read ratings-result file to get the ratings for each participant and for each image
    rating_path = '../data/ratings-results/ratings-results.csv'
    data = pd.read_csv(rating_path, index_col=0)

    # exclude participant with bad acquisitions
    data = data[(data['code'] != 'maba09') & (data['code'] != 'soze31') & (data['code'] != 'nipe10') &
                (data['code'] != 'dino02') & (data['code'] != 'krki20')]

    # format the data
    data['code'] = data['code'].astype('category')
    data['img_name'] = data['img_name'].astype('category')
    data['manipulation'] = data['manipulation'].astype('category')

    # ----------------------------------------------------------------------------------------------------------------
    # h1: original vs new ratings

    # get the list of original images
    image_codes = data.loc[:, 'img_name'].to_list()
    image_codes = list(set(image_codes))
    image_codes = [code for code in image_codes if '_orig' in code]

    features = [['vm', 'valence'], ['am', 'arousal']]

    # extract old and new mean ratings for each image
    for feature in features:
        x = data.loc[data['img_name'].isin(image_codes)]
        data_wide = x.groupby(['img_name']).mean()[feature].dropna().reset_index()
        data_wide.to_csv('../data/jamovi-tables/h1_' + feature[1] + '.csv', index=False)

    # ----------------------------------------------------------------------------------------------------------------
    # h2: subject ID and rating according to the manipulation

    # get the list of participants
    participant_codes = data.loc[:, 'code'].to_list()
    participant_codes = list(set(participant_codes))

    features = ['valence', 'arousal']

    for feature in features:
        # get the mean ratings for each participant according to the type of manipulation
        data_wide = data.groupby(['code', 'manipulation']).mean()[feature].reset_index().pivot(index='code',
                                                                                               columns=['manipulation'])
        data_wide.columns = ["_".join(col[1:]).strip() for col in data_wide.columns.values]

        # get participant gender
        gender = []
        for idx, row in data_wide.iterrows():
            gender.append(participant_data.loc[participant_data['code'] == idx]['gender'].values[0][0])

        # refine dataframe and save it
        data_wide.insert(0, 'gender', gender)
        data_wide['gender'] = data_wide['gender'].astype('category')
        data_wide.to_csv('../data/jamovi-tables/h2_' + feature + '.csv', index=False)

    # ----------------------------------------------------------------------------------------------------------------
    # h3: subject ID and N200 amplitude according to the manipulation

    path = '../data/pickle/'
    dict_info = json.load(open("../data/eeg/info.json"))

    # prepare the two dataframe to contain data
    n200_wide = pd.DataFrame(columns=['gender', 'blackwhite', 'blurring', 'circularblurringext',
                                      'circularblurringint', 'edges', 'original'])
    p300_wide = n200_wide.copy()
    columns = list(n200_wide.columns)[2:]

    # for each participant
    for code in participant_codes:

        # extract its gender and process its file
        g = participant_data.loc[participant_data['code'] == code]['gender'].values[0][0]

        file_types = ['data', 'labels', 'info']
        eeg = {}
        for type in file_types:
            file_path = path + code + '_' + type + '.pkl'
            with open(file_path, 'rb') as f:
                eeg[type] = pickle.load(f)

        peaks = get_peak_pickle(epochs=eeg[file_types[0]], channels_list=eeg[file_types[2]]['channels'],
                                channels_interest=['TP9', 'FT9', 'T7'], labels=eeg['labels'],
                                t_min_epoch=eeg[file_types[2]]['tmin'], fs=eeg[file_types[2]]['fs'],
                                t_min=0.180, t_max=0.250, peak=-1)

        peaks_values = [code, g] + [peaks[key] for key in columns]
        n200_wide.loc[len(n200_wide.index)] = peaks_values

        peaks = get_peak_pickle(epochs=eeg[file_types[0]], channels_list=eeg[file_types[2]]['channels'],
                                channels_interest=['P3', 'Pz', 'P4'], labels=eeg['labels'],
                                t_min_epoch=eeg[file_types[2]]['tmin'], fs=eeg[file_types[2]]['fs'],
                                t_min=0.280, t_max=0.350, peak=1)

        peaks_values = [g] + [peaks[key] for key in columns]
        p300_wide.loc[len(p300_wide.index)] = peaks_values

    n200_wide['code'] = n200_wide['code'].astype('category')
    n200_wide['gender'] = n200_wide['gender'].astype('category')
    n200_wide.to_csv('../data/jamovi-tables/h3_n200.csv', index=False)

    p300_wide['code'] = p300_wide['code'].astype('category')
    p300_wide['gender'] = p300_wide['gender'].astype('category')
    p300_wide.to_csv('../data/jamovi-tables/h3_p300.csv', index=False)

    # ----------------------------------------------------------------------------------------------------------------
    # h4 - n200 vs valence

    path = '../data/pickle/'
    dict_info = json.load(open('../data/eeg/info_full.json'))

    n200_wide = pd.DataFrame(columns=['code', 'gender', 'blackwhite', 'blurring', 'circularblurringext',
                                      'circularblurringint', 'edges', 'original'])
    columns = list(n200_wide.columns)[2:]

    n200_wide, p300_wide = {}, {}
    for column in columns:
        n200_wide[column] = pd.DataFrame(columns=['n200', 'valence'])
        p300_wide[column] = pd.DataFrame(columns=['p300', 'valence'])

    for code in participant_codes:

        file_path = path + 'subj_' + code + '_block1.xdf'
        g = participant_data.loc[participant_data['code'] == code]['gender'].values[0][0]

        file_types = ['data', 'labels', 'info']
        eeg = {}
        for type in file_types:
            file_path = path + code + '_' + type + '.pkl'
            with open(file_path, 'rb') as f:
                eeg[type] = pickle.load(f)

        n200_peaks, n200_annotations = get_peak_pickle(epochs=eeg[file_types[0]],
                                                       channels_list=eeg[file_types[2]]['channels'],
                                                       channels_interest=['TP9', 'FT9', 'T7'], labels=eeg['labels'],
                                                       t_min_epoch=eeg[file_types[2]]['tmin'],
                                                       fs=eeg[file_types[2]]['fs'],
                                                       t_min=0.180, t_max=0.250, peak=-1, mean=False)

        p300_peaks, p300_annotations = get_peak_pickle(epochs=eeg[file_types[0]],
                                                       channels_list=eeg[file_types[2]]['channels'],
                                                       channels_interest=['P3', 'Pz', 'P4'], labels=eeg['labels'],
                                                       t_min_epoch=eeg[file_types[2]]['tmin'],
                                                       fs=eeg[file_types[2]]['fs'],
                                                       t_min=0.280, t_max=0.350, peak=1, mean=False)

        for condition in n200_wide.keys():

            peaks = n200_peaks[condition]
            annotations = n200_annotations[condition]

            ratings = [float(data.loc[(data['code'] == code) & (data['img_name'] == annotation)]['valence'])
                       for annotation in annotations]

            values = np.vstack((np.array(peaks), np.array(ratings))).T

            for value in values:
                n200_wide[condition].loc[len(n200_wide[condition].index)] = value

            peaks = p300_peaks[condition]
            annotations = p300_annotations[condition]

            ratings = [float(data.loc[(data['code'] == code) & (data['img_name'] == annotation)]['valence'])
                       for annotation in annotations]

            values = np.vstack((np.array(peaks), np.array(ratings))).T

            for value in values:
                p300_wide[condition].loc[len(p300_wide[condition].index)] = value

    for condition in n200_wide.keys():
        n200_wide[condition].to_csv('../data/jamovi-tables/h4_n200_' + condition + '.csv', index=False)
        p300_wide[condition].to_csv('../data/jamovi-tables/h4_p300_' + condition + '.csv', index=False)
