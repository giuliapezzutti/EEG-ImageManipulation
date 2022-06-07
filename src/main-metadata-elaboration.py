import glob
import re

import pandas as pd
from functions import *


if __name__ == '__main__':

    # unique filepath for the form results file
    filepath = '../data/form-results/Form results - Risposte del modulo 1.csv'
    output_folder = '../data/form-results/'

    # reading of the file
    df = pd.read_csv(filepath)

    # extraction of the participant codes and columns names
    codes = df.iloc[:]['INSERT YOUR PARTICIPANT CODE'].tolist()
    cols = list(df.columns)

    # determination of the columns of interest (demographic information) and removal of non-interesting parts
    columns = df.columns[1:6]
    columns_list = list(columns)
    for idx, name in enumerate(columns_list):
        name = name.lower()
        columns_list[idx] = name.replace('insert your ', '')

    # definition of the types of personality according to the definition
    p_types = ['E', 'A', 'C', 'N', 'O']

    # creation of the new dataframe with the columns of interest
    new_columns = columns_list + p_types
    participants_results = pd.DataFrame(columns=new_columns)

    # cycle for each participant
    for code in codes:

        # extraction data corresponding to the participant of interest
        data = df.loc[df['INSERT YOUR PARTICIPANT CODE'] == code]

        # extraction of the data in the columns of interest
        results = list(data[columns].values[0])
        results = list(map(lambda x: str(x).lower(), results))

        # extraction of the columns of interest in the personality computation: all the ones that starts with 'I...'
        p = []
        for col in cols:
            res = re.findall("^I[.]{3} [\[]{1}.+[\]]{1}", str(col))
            if res:
                p.append(res[0])

        # extraction of the data regarding the columns of interest just extracted and cast to int values
        p_data = list(data.iloc[0][p])
        p_data = list(int(vote[0]) for vote in p_data)

        # creation of the personality matrix and bias vector
        p_matrix, p_bias = create_personality_matrix(len(p_types), len(p_data), p_types)

        # computation for the derivation of each personality score
        p_results = list(np.array(p_matrix.dot(p_data) + p_bias, dtype='int'))

        # concatenation of data of interest and saving of it inside the final dataframe
        results = results + p_results
        participants_results.loc[participants_results.shape[0]] = results

    # saving of the results regarding the form
    participants_results = participants_results.rename(columns={'participant code': 'code'})
    participants_results.to_csv(output_folder+'/form-results.csv')

    # COMPUTATION OF ANALYSIS FOR ONLINE IMAGE EVALUATION

    # path to the folder containing the csv files results
    rating_path = '../data/ratings-results/'
    available_ratings_paths = glob.glob(rating_path + '/*.csv')

    # definition of the columns of interest, for the input file (columns) and for the output file (new_columns)
    columns = list(['img_name', 'manipulation', 'vm', 'am', 'valence_slider.response', 'arousal_slider.response'])
    participant_code = 'participant code'
    new_columns = [participant_code] + columns

    # creation of results dataframe
    responses = pd.DataFrame(columns=new_columns)

    mean_ratings = {}
    # cycling over all participants codes (found in online form)
    for code in codes:

        # find the filepath (if exists) containing the participant code
        correct = None
        for filepath in available_ratings_paths:
            if filepath.find(code) != -1:
                correct = filepath
        if correct is None:
            continue

        # reading dataframe
        df = pd.read_csv(correct)

        short_names = []
        image_names = df.loc[:, 'img_name']
        original_image_names = [name for name in image_names if '_orig' in name]
        original_image_names = list(set(original_image_names))

        for name in original_image_names:
            image_ratings = df.loc[df['img_name'] == name, ['valence_slider.response', 'arousal_slider.response']].values.tolist()[0]
            label = name.rsplit('_', 1)[0]

            if label in mean_ratings:
                mean_ratings[label] = np.concatenate((mean_ratings[label], np.array([image_ratings])))
            else:
                mean_ratings[label] = np.array([image_ratings])

        # extraction of the columns of interest and subsequent saving
        data = df[columns]
        # data = data.round({'valence_slider.response': 0, 'arousal_slider.response': 0})
        data.insert(0, participant_code, code)

        means = data.loc[:, ['vm', 'am']]
        new_ratings = data.loc[:, ['valence_slider.response', 'arousal_slider.response']]
        labels, valence_difference, arousal_difference = [], [], []
        threshold = 0.1

        for index, row in means.iterrows():
            valence = row['vm']
            arousal = row['am']

            if (np.square(valence) + np.square(arousal)) <= np.square(threshold):
                labels.append('Neutral')
            elif valence > 0 and arousal > 0:
                labels.append('HVHA')
            elif valence > 0 and arousal <= 0:
                labels.append('HVLA')
            elif valence <= 0 and arousal > 0:
                labels.append('LVHA')
            elif valence <= 0 and arousal <= 0:
                labels.append('LVLA')

        data.insert(data.shape[1], 'label', labels, True)

        responses = pd.concat([responses, data])

    for image in mean_ratings.keys():
        mean_ratings[image] = np.mean(mean_ratings[image], axis=0)

    responses = responses.rename(columns={'participant code': 'code',
                                          'valence_slider.response': 'valence', 'arousal_slider.response': 'arousal'})

    # saving of the csv file containing all the data
    responses.to_csv(rating_path+'/ratings-results.csv')

    conditions = responses.loc[:, 'manipulation'].values.tolist()
    conditions = list(set(conditions))
