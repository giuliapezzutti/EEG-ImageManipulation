def define_ers_erd(self):
    def square_signal(array_data):
        return np.square(array_data)

    # x_axis = list(range(int(self.t_min * 1000 - 2), int(self.t_max * 1000 + 1), 2))
    x_axis = None

    f_min = int(self.input_info['erds'][0])
    f_max = int(self.input_info['erds'][1])

    f_start = list(range(f_min, f_max, 1))
    f_end = list(range(f_min + 2, f_max + 2, 1))
    f_plot = list(range(f_min + 1, f_max + 2, 1))

    signals = self.raw.copy()
    filter_bank = []

    # for each frequency band in the erds
    for idx, start in enumerate(f_start):
        # filter the signal
        filtered_signals = signals.filter(start, f_end[idx], l_trans_bandwidth=1, h_trans_bandwidth=1)

        # divide into epochs
        filtered_signals.set_annotations(self.annotations)
        events, event_mapping = mne.events_from_annotations(self.raw)
        epochs = mne.Epochs(filtered_signals, events, self.event_mapping, preload=True, baseline=(self.t_min, 0),
                            reject=self.input_info['epochs_reject_criteria'], tmin=self.t_min, tmax=self.t_max)

        # save the obtained epochs
        filter_bank.append(epochs)

    # generation of the erds images, one for each condition and for each roi

    # for each type of epoch
    for condition in self.event_mapping.keys():

        print(condition)

        # for each roi
        for roi, roi_numbers in self.rois_numbers.items():

            freq_erds_results = []

            # for each frequency band (so for each set of epochs previously saved)
            for freq_band_epochs in filter_bank:

                # take frequency band of interest
                condition_epochs = freq_band_epochs[condition].copy()

                # take channels of interest
                condition_epochs = condition_epochs.pick(roi_numbers)

                # square each epoch
                condition_epochs = condition_epochs.apply_function(square_signal)

                # extract data
                epochs_data = condition_epochs.get_data()

                if x_axis is None:
                    x_axis = condition_epochs.times
                    x_axis = x_axis * 1000
                    step = np.abs(x_axis[1] - x_axis[0])
                    x_axis = np.append(x_axis, x_axis[-1] + step)
                    x_axis = np.array(x_axis, dtype=int)

                # derive reference for each epoch and channel
                reference = epochs_data[:, :, :int(self.t_min * self.eeg_fs)]
                reference_power = np.mean(reference, axis=2)

                # for each value inside the data, compute the ERDS value -> trial-individual references
                erds_epochs = []
                for idx_epoch, epoch in enumerate(epochs_data):
                    for idx_ch, channel in enumerate(epoch):
                        erds = np.zeros(epochs_data.shape[2])
                        for sample, power in enumerate(channel):
                            current_reference_power = reference_power[idx_epoch, idx_ch]
                            erds[sample] = (power - current_reference_power) / current_reference_power * 100
                        erds_epochs.append(erds)

                # mean for each epoch and channel and save
                mean_erds = np.mean(np.array(erds_epochs), axis=0)
                freq_erds_results.append(mean_erds)

            # visualization
            freq_erds_results = np.array(freq_erds_results)
            z_min, z_max = -np.abs(freq_erds_results).max(), np.abs(freq_erds_results).max()
            fig, ax = plt.subplots()
            p = ax.pcolor(x_axis, f_plot, freq_erds_results, cmap='RdBu', snap=True, vmin=z_min, vmax=z_max)
            ax.set_xlabel('Time (\u03bcs)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(condition + ' ' + roi)
            ax.axvline(0, color='k')
            fig.colorbar(p, ax=ax)
            fig.savefig(self.file_info['output_folder'] + '/' + condition + '_' + roi + '_erds.png')
            plt.close()