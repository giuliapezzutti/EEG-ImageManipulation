import glob
import pickle

import pandas as pd

if __name__ == '__main__':

    paths = glob.glob('../data/pickle/*.pkl')
    paths = [path.replace('\\', '/') for path in paths if '_data' in path]

    codes = [(path.rsplit('_', 1)[0]).rsplit('/', 1)[1] for path in paths]

    participant_data = pd.read_csv('../data/form-results/form-results.csv')
    participant_data = participant_data[['code', 'gender']]

    for idx, code in enumerate(codes):

        with open(paths[idx], 'rb') as f:
            data = pickle.load(f)
        with open(paths[idx].replace('_data', '_info'), 'rb') as f:
            info = pickle.load(f)

        g = participant_data.loc[participant_data['code'] == code]['gender'].values[0]
