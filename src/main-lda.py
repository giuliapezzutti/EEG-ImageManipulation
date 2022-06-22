import glob
import json
import pickle
import re

import mne
import numpy as np
import pandas as pd
from scipy import signal
import seaborn
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from src.functions import get_label


def bandpower(x, fs, fmin, fmax):
    f, Pxx = signal.periodogram(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])


def lda_model(X_train, y_train, X_test, y_test):

    model = LDA()
    model.fit(X_train, y_train)

    print('Valence')
    print('Train score:', model.score(X_train, y_train))
    print('Test score:', model.score(X_test, y_test))

    return model.coef_


def lr_model(X_train, y_train, X_test, y_test):

    from torch.autograd import Variable

    class LogisticRegression(torch.nn.Module):
        def __init__(self, inputSize, outputSize):
            super(LogisticRegression, self).__init__()
            self.linear = torch.nn.Linear(inputSize, outputSize)
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            out = self.linear(x)
            out = self.sigmoid(out)
            return out

    inputDim = X.shape[1]  # takes variable 'x'
    outputDim = Y.shape[1]  # takes variable 'y'
    learningRate = 0.05
    number_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LogisticRegression(inputDim, outputDim)
    model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    ltr_curve, atr_curve, lts_curve, ats_curve = [], [], [], []

    for epoch in range(number_epochs):
        # Train Set
        inputs = Variable(torch.tensor(X_train.values.astype(np.float32))).to(device)
        labels = Variable(torch.tensor(y_train.astype(np.float32))).to(device)

        optimizer.zero_grad()
        y_hat = model(inputs)

        labels = np.squeeze(labels)
        y_hat = np.squeeze(y_hat)

        loss = criterion(y_hat, labels)
        train_loss = loss.item()
        ltr_curve.append(train_loss)

        train_acc = torch.sum(y_hat.round() == labels) / labels.size(0)
        atr_curve.append(train_acc.cpu().detach().numpy())

        # Optimization
        loss.backward()
        optimizer.step()

        # Test set
        y_hat = model(Variable(torch.tensor(X_test.values.astype(np.float32))).to(device))
        test_lbls = Variable(torch.tensor(y_test.astype(np.float32))).to(device)

        test_lbls = np.squeeze(test_lbls)
        y_hat = np.squeeze(y_hat)

        test_loss = criterion(y_hat, test_lbls)
        test_loss = test_loss.item()
        lts_curve.append(test_loss)
        test_acc = torch.sum(y_hat.round() == test_lbls) / test_lbls.size(0)
        ats_curve.append(test_acc.cpu().detach().numpy())

        print('epoch {}, loss {}, test accuracy {}'.format(epoch, loss.item(), test_acc))

    print('Final test accuracy: ', test_acc)

    fig = plt.figure(figsize=(16, 10))
    plt.plot(ltr_curve, label="Loss train")
    plt.plot(atr_curve, label="Acc  train")
    plt.plot(lts_curve, label="Loss test")
    plt.plot(ats_curve, label="Acc  test")
    plt.legend()
    # plt.show()
    plt.close()

    coef = [p.cpu().detach().numpy() for p in model.parameters()]

    return coef


if __name__ == '__main__':

    # list of paths in pickle folder
    paths = glob.glob('../data/pickle/*.pkl')
    paths = [path.replace('\\', '/') for path in paths if '_data' in path]

    # list of codes for which pickles are available
    codes = [(path.rsplit('_', 1)[0]).rsplit('/', 1)[1] for path in paths]

    # loading of main information
    input_info = json.load(open('../data/eeg/info.json'))
    rois = input_info['rois']

    participant_data = pd.read_csv('../data/form-results/form-results.csv')
    participant_data = participant_data[['code', 'gender']]

    ratings_data = pd.read_csv('../data/ratings-results/ratings-results.csv')
    ratings_data = ratings_data[['code', 'img_name', 'valence', 'arousal']]

    pd_signal = None
    pd_gender, coefs_valence, coefs_arousal = [], [], []

    for idx, code in enumerate(codes):

        if code == 'krki20' or code == 'nipe10':
            continue

        print(code)

        with open(paths[idx], 'rb') as f:
            data = pickle.load(f)
        with open(paths[idx].replace('_data', '_info'), 'rb') as f:
            info = pickle.load(f)
            index_channels_eeg = [index for index in range(len(info['channels'])) if 'EOG' not in info['channels'][index]]
            channels_eeg = np.array(info['channels'])[index_channels_eeg]
        with open(paths[idx].replace('_data', '_labels'), 'rb') as f:
            labels = pickle.load(f)
            conditions = [label.split('/')[1] for label in labels]
            unique_conditions = list(set(conditions))

        # remove EOG data
        data = data[:, index_channels_eeg]

        # get gender of the current participant
        g = participant_data.loc[participant_data['code'] == code]['gender'].values[0]

        # flatten each epoch: #epochs x (#channels*#samples)
        epochs = data.reshape(data.shape[0], -1)
        gender = np.expand_dims(np.array([g] * (epochs.shape[0])), axis=-1)

        # concatenate the epochs to what already present
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
        frontal_amplitude = np.mean(frontal_amplitude, axis=-1) * 1e6

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
            peaks.append(np.min(peak_mag) * 1e6)

        # calculate mean gamma power

        powers = []
        for epoch in data:
            epoch_powers = []
            for channel in epoch:
                epoch_powers.append(bandpower(channel, info['fs'], 30, 100))
            powers.append(np.mean(np.array(epoch_powers)))

        # calculate correspondent label

        responses = []
        for label in labels:
            img_name = label.split('/')[0]
            valence, arousal = ratings_data.loc[ratings_data['code'] == code].loc[ratings_data['img_name'] == img_name][['valence', 'arousal']].values[0]
            responses.append(get_label(valence, arousal))

        valence = [response[0] for response in responses]
        arousal = [response[2] for response in responses]

        # VALENCE dataframe

        pd_data_valence = np.vstack((np.array(valence), np.array(frontal_amplitude, dtype=float),
                                     np.array(peaks, dtype=float), np.array(powers, dtype=float))).T
        pd_data_valence = pd.DataFrame(data=pd_data_valence, columns=['valence', 'f-amp', 'tl-peak', 'gamma-power'])
        pd_data_valence[['f-amp', 'tl-peak', 'gamma-power']] = pd_data_valence[['f-amp', 'tl-peak', 'gamma-power']].astype(float)

        X = pd_data_valence[['f-amp', 'tl-peak', 'gamma-power']]
        Y = pd_data_valence[['valence']] == 'H'

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)

        # VALENCE LDA

        print('\nValence prediction with LDA')

        coefs = lda_model(X_train, y_train, X_test, y_test)

        coefs_valence.append(coefs[0])
        print(coefs)
        # sns.pairplot(pd_data_valence, hue='valence')

        # VALENCE LOGISTIC REGRESSION

        print('\nValence prediction with LR')

        coefs = lr_model(X_train, y_train, X_test, y_test)
        print(coefs)

        # AROUSAL dataframe

        pd_data_arousal = np.vstack((np.array(arousal), frontal_amplitude, np.array(peaks), np.array(powers))).T
        pd_data_arousal = pd.DataFrame(data=pd_data_arousal, columns=['arousal', 'f-amp', 'tl-peak', 'gamma-power'])
        pd_data_arousal[['f-amp', 'tl-peak', 'gamma-power']] = pd_data_arousal[['f-amp', 'tl-peak', 'gamma-power']].astype(float)

        X = pd_data_arousal[['f-amp', 'tl-peak', 'gamma-power']]
        Y = pd_data_arousal[['arousal']] == 'H'

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)

        # AROUSAL LDA

        print('\nArousal prediction with LDA')

        coefs = lda_model(X_train, y_train, X_test, y_test)

        coefs_arousal.append(coefs[0])
        print(coefs)
        sns.pairplot(pd_data_arousal, hue='arousal')

        # AROUSAL LOGISTIC REGRESSION

        print('\nArousal prediction with LR')

        coefs = lr_model(X_train, y_train, X_test, y_test)
        print(coefs)

        print('\n')

    coefs_valence = np.array(coefs_valence)
    coefs_arousal = np.array(coefs_arousal)

    print('Mean coefficients for valence', np.mean(coefs_valence, axis=0))
    print('Mean coefficients for arousal', np.mean(coefs_arousal, axis=0))

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

    coefs = pd.DataFrame(data=coefs, index=channels_eeg)

    seaborn.heatmap(coefs)
    plt.savefig('../images/lda/coefs_sex_classification.png')
    plt.show()
