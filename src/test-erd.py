import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import seaborn as sns
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test

fnames = eegbci.load_data(subject=1, runs=[6, 10, 14])
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in fnames])

raw.rename_channels(lambda x: x.strip('.'))  # remove dots from channel names

events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))

tmin, tmax = -1, 4
event_ids = dict(hands=2, feet=3)  # map event IDs to tasks

epochs = mne.Epochs(raw, events, event_ids, tmin - 0.5, tmax + 0.5,
                    picks=('C3', 'Cz', 'C4'), baseline=None, preload=True)

freqs = np.arange(2, 36)  # frequencies from 2-35Hz
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
baseline = [-1, 0]  # baseline interval (in s)
cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
              buffer_size=None, out_type='mask')  # for cluster test

tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs, use_fft=True,
                     return_itc=False, average=False, decim=2)
tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")

for event in event_ids:
    # select desired epochs for visualization
    tfr_ev = tfr[event]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [10, 10, 10, 1]})
    for ch, ax in enumerate(axes[:-1]):  # for each channel
        # positive clusters
        _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
        # negative clusters
        _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)

#         # note that we keep clusters with p <= 0.05 from the combined clusters
#         # of two independent tests; in this example, we do not correct for
#         # these two comparisons
#         c = np.stack(c1 + c2, axis=2)  # combined clusters
#         p = np.concatenate((p1, p2))  # combined p-values
#         mask = c[..., p <= 0.05].any(axis=-1)
#
#         # plot TFR (ERDS map with masking)
#         tfr_ev.average().plot([ch], cmap="RdBu", cnorm=cnorm, axes=ax,
#                               colorbar=False, show=False, mask=mask,
#                               mask_style="mask")
#
#         ax.set_title(epochs.ch_names[ch], fontsize=10)
#         ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
#         if ch != 0:
#             ax.set_ylabel("")
#             ax.set_yticklabels("")
#     fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
#     fig.suptitle(f"ERDS ({event})")
#     plt.show()
