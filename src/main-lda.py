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

from sklearn.model_selection import train_test_split, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from src.functions import get_label


def bandpower(x, fs, fmin, fmax):
    f, Pxx = signal.periodogram(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])


def lda_model(X_train, y_train, X_test, y_test, return_coef=True):

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    model = LDA()
    model.fit(X_train, y_train)

    if return_coef:
        return model.coef_
    return model.score(X_test, y_test)


def lr_model_train(X_train, y_train, X_test, y_test, return_coef=True):

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
        inputs = Variable(torch.tensor(X_train.astype(np.float32))).to(device)
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
        y_hat = model(Variable(torch.tensor(X_test.astype(np.float32))).to(device))
        test_lbls = Variable(torch.tensor(y_test.astype(np.float32))).to(device)

        test_lbls = np.squeeze(test_lbls)
        y_hat = np.squeeze(y_hat)

        test_loss = criterion(y_hat, test_lbls)
        test_loss = test_loss.item()
        lts_curve.append(test_loss)
        test_acc = torch.sum(y_hat.round() == test_lbls) / test_lbls.size(0)
        ats_curve.append(test_acc.cpu().detach().numpy())

        # print('epoch {}, loss {}, test accuracy {}'.format(epoch, loss.item(), test_acc))

    fig = plt.figure(figsize=(16, 10))
    plt.plot(ltr_curve, label="Loss train")
    plt.plot(atr_curve, label="Acc  train")
    plt.plot(lts_curve, label="Loss test")
    plt.plot(ats_curve, label="Acc  test")
    plt.legend()
    # plt.show()
    plt.close()

    coef = [p.cpu().detach().numpy() for p in model.parameters()]

    if return_coef:
        return coef
    return lts_curve[-1]


def lr_training_Kfold(train_function, X, Y):

    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    cvl = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index, :], X[test_index, :]
        Y_train, Y_test = Y[train_index, :], Y[test_index, :]
        cvl.append(train_function(X_train, Y_train, X_test, Y_test, return_coef=False))

    print('Mean accuracy:', np.mean(np.array(cvl)))
    print('Std accuracy:', np.std(np.array(cvl)))

    # plt.plot(cvl)
    # plt.show()
    #
    # sns.kdeplot(cvl)
    # plt.show()

    return cvl


def lr_chance_level(train_function, X, Y, cross_validation_scores):

    N_perm = 5
    chance_cvl = []

    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    for i in range(N_perm):
        cvl = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index, :], X[test_index, :]
            Y = np.random.permutation(Y)
            Y_train, Y_test = Y[train_index, :], Y[test_index, :]
            cvl.append(train_function(X_train, Y_train, X_test, Y_test, return_coef=False))

        # plt.plot(cvl)
        # plt.show()
        print(f"Training {i}, accuracy: {cvl[-1]}")
        chance_cvl.append(cvl)

    # for i in chance_cvl:
    #     sns.kdeplot(i)
    # sns.kdeplot(cross_validation_scores, label="Accuracy scores")
    # plt.show()

    return chance_cvl


def test_lda_lr(label, X, Y):

    # LINEAR DISCRIMINANT ANALYSIS

    print('\n'+label+' prediction with LDA')

    lda = lr_training_Kfold(lda_model, X.to_numpy(), Y.to_numpy())
    lda_chance = lr_chance_level(lda_model, X.to_numpy(), Y.values.astype(int), lda)

    # LOGISTIC REGRESSION

    print('\n'+label+' prediction with LR')

    lr = lr_training_Kfold(lr_model_train, X.to_numpy(), Y.to_numpy())
    lr_chance = lr_chance_level(lr_model_train, X.to_numpy(), Y.values.astype(int), lr)

    return lda, lda_chance, lr, lr_chance


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

    valence_lda, valence_lr, arousal_lda, arousal_lr = [], [], [], []
    valence_lda_chance, valence_lr_chance, arousal_lda_chance, arousal_lr_chance = [], [], [], []

    signal_valence_lda, signal_valence_lr, signal_arousal_lda, signal_arousal_lr = [], [], [], []
    signal_valence_lda_chance, signal_valence_lr_chance, signal_arousal_lda_chance, signal_arousal_lr_chance = [], [], [], []

    codes.remove('krki20')
    print(codes)

    for idx, code in enumerate(codes):

        print(code)

        path = [i for i in paths if code in i]
        path = [i for i in path if '_data' in i][0]

        with open(path, 'rb') as f:
            data = pickle.load(f)
        with open(path.replace('_data', '_info'), 'rb') as f:
            info = pickle.load(f)
            index_channels_eeg = [index for index in range(len(info['channels'])) if 'EOG' not in info['channels'][index]]
            channels_eeg = np.array(info['channels'])[index_channels_eeg]
        with open(path.replace('_data', '_labels'), 'rb') as f:
            labels = pickle.load(f)
            if len(labels[0].split('/')) > 1:
                conditions = [label.split('/')[1] for label in labels]
            else:
                conditions = labels
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

        # calculate correspondent label

        responses = []
        for label in labels:
            img_name = label.split('/')[0]
            valence, arousal = ratings_data.loc[ratings_data['code'] == code].loc[ratings_data['img_name'] == img_name][
                        ['valence', 'arousal']].values[0]
            responses.append(get_label(valence, arousal))

        valence = [response[0] for response in responses]
        arousal = [response[2] for response in responses]

        # -------------------------------------------------------------------------------------------------------------

        # print('Signal analysis\n')
        #
        # # VALENCE signal
        #
        # pd_data_valence = pd.DataFrame(data=epochs)
        # pd_data_valence['valence'] = valence
        #
        # X = pd_data_valence.drop('valence', 1)
        # Y = pd_data_valence[['valence']] == 'H'
        #
        # lda, lda_chance, lr, lr_chance = test_lda_lr('valence', X, Y)
        # signal_valence_lda.append(lda)
        # signal_valence_lda_chance.append(lda_chance)
        # signal_valence_lr.append(lr)
        # signal_valence_lr_chance.append(lr_chance)
        #
        # # VALENCE signal
        #
        # pd_data_valence = pd.DataFrame(data=epochs)
        # pd_data_valence['arousal'] = arousal
        #
        # X = pd_data_valence.drop('arousal', 1)
        # Y = pd_data_valence[['arousal']] == 'H'
        #
        # lda, lda_chance, lr, lr_chance = test_lda_lr('arousal', X, Y)
        # signal_arousal_lda.append(lda)
        # signal_arousal_lda_chance.append(lda_chance)
        # signal_arousal_lr.append(lr)
        # signal_arousal_lr_chance.append(lr_chance)

        # -------------------------------------------------------------------------------------------------------------

        print('\nFeatures analysis')

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

        # -------------------------------------------------------------------------------------------------------------

        # VALENCE features

        pd_data_valence = np.vstack((np.array(valence), np.array(frontal_amplitude, dtype=float),
                                     np.array(peaks, dtype=float), np.array(powers, dtype=float))).T
        pd_data_valence = pd.DataFrame(data=pd_data_valence, columns=['valence', 'f-amp', 'tl-peak', 'gamma-power'])
        pd_data_valence[['f-amp', 'tl-peak', 'gamma-power']] = pd_data_valence[['f-amp', 'tl-peak', 'gamma-power']].astype(float)

        X = pd_data_valence[['f-amp', 'tl-peak', 'gamma-power']]
        Y = pd_data_valence[['valence']] == 'H'

        lda, lda_chance, lr, lr_chance = test_lda_lr('valence', X, Y)
        valence_lda.append(lda)
        valence_lda_chance.append(lda_chance)
        valence_lr.append(lr)
        valence_lr_chance.append(lr_chance)

        # AROUSAL features

        pd_data_arousal = np.vstack((np.array(arousal), frontal_amplitude, np.array(peaks), np.array(powers))).T
        pd_data_arousal = pd.DataFrame(data=pd_data_arousal, columns=['arousal', 'f-amp', 'tl-peak', 'gamma-power'])
        pd_data_arousal[['f-amp', 'tl-peak', 'gamma-power']] = pd_data_arousal[['f-amp', 'tl-peak', 'gamma-power']].astype(float)

        X = pd_data_arousal[['f-amp', 'tl-peak', 'gamma-power']]
        Y = pd_data_arousal[['arousal']] == 'H'

        lda, lda_chance, lr, lr_chance = test_lda_lr('arousal', X, Y)
        arousal_lda.append(lda)
        arousal_lda_chance.append(lda_chance)
        arousal_lr.append(lr)
        arousal_lr_chance.append(lr_chance)

        print('\n')

    cvl_scores = [valence_lda, valence_lr, arousal_lda, arousal_lr]
    cvl_scores_chance = [valence_lda_chance, valence_lr_chance, arousal_lda_chance, arousal_lr_chance]
    cvl_scores_names = ['valence_lda', 'valence_lr', 'arousal_lda', 'arousal_lr']

    for model_accuracies, chance_accuracies, name in zip(cvl_scores, cvl_scores_chance, cvl_scores_names):

        print('\n', name)

        cvl = np.matrix(model_accuracies)
        mean = np.array(np.mean(cvl, axis=1)).flatten()
        std = np.array(np.std(cvl, axis=1)).flatten()
        x_pos = np.arange(len(mean))

        print(np.mean(np.array(chance_accuracies)))
        percentile = np.percentile(np.array(chance_accuracies).flatten(), [100*(1-0.95)/2, 100*(1-(1-0.95)/2)])[1]
        print(percentile)

        chance_accuracies = np.array(chance_accuracies).flatten()
        np.save('../images/lda/chance_values_'+name+'.npy', chance_accuracies)
        sns.kdeplot(chance_accuracies)
        plt.show()

        fig, ax = plt.subplots()
        ax.bar(x_pos, list(mean), yerr=list(std), align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.axhline(percentile, x_pos[0], x_pos[-1])
        ax.set_ylabel('Accuracy')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(codes, rotation='vertical')
        ax.set_title(name)
        plt.tight_layout()
        plt.savefig('../images/lda/features_'+name+'.png')
        plt.close()

    # cvl_scores = [signal_valence_lda, signal_valence_lr, signal_arousal_lda, signal_arousal_lr]
    # cvl_scores_chance = [signal_valence_lda_chance, signal_valence_lr_chance, signal_arousal_lda_chance, signal_arousal_lr_chance]
    # cvl_scores_names = ['valence_lda', 'valence_lr', 'arousal_lda', 'arousal_lr']
    #
    # for model_accuracies, chance_accuracies, name in zip(cvl_scores, cvl_scores_chance, cvl_scores_names):
    #
    #     cvl = np.matrix(model_accuracies)
    #     mean = np.array(np.mean(cvl, axis=1)).flatten()
    #     std = np.array(np.std(cvl, axis=1)).flatten()
    #     x_pos = np.arange(len(mean))
    #
    #     print(chance_accuracies)
    #     percentile = np.percentile(np.array(chance_accuracies).flatten(), [100*(1-0.95)/2, 100*(1-(1-0.95)/2)])[1]
    #     print(percentile)
    #
    #     fig, ax = plt.subplots()
    #     ax.bar(x_pos, list(mean), yerr=list(std), align='center', alpha=0.5, ecolor='black', capsize=10)
    #     ax.axhline(percentile, x_pos[0], x_pos[-1])
    #     ax.set_ylabel('Accuracy')
    #     ax.set_xticks(x_pos)
    #     ax.set_xticklabels(codes, rotation='vertical')
    #     ax.set_title(name)
    #     plt.tight_layout()
    #     plt.savefig('../images/lda/signal_'+name+'.png')
    #     plt.close()

    exit(1)

    # -----------------------------------------------------------------------------------------------------------------

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
