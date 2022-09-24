import mne
from os.path import join
import numpy as np
from dPTE import epo_dPTE
from scipy.signal import find_peaks
from scipy.stats import zscore
import matplotlib.pyplot as plt
plt.ion()
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_sensors_connectivity


root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
data_dir = join(root_dir, mem_dir, "02_MemTask")

frontal_chans = ["F3", "F1", "Fz", "F2", "F4", "FC3", "FC1", "FC2", "FC4"]
parietal_chans = ["PO7", "PO3", "POz", "PO4", "PO8", "P5", "P1", "P2", "P6"]
freqs = [4., 4.5, 5., 5.5, 6., 6.5]

subjs = ["120"]
sessions = ["2"]
preposts = ["pre", "post"]

for subj in subjs:
    subj_str = f"MT-YG-{subj}"
    subj_dir = join(data_dir, subj_str)
    for sess in sessions:
        for pp in preposts:
            sess_dir = join(subj_dir, f"Session{sess}", "EEG")
            # raw = mne.io.Raw(join(sess_dir, f"MT-YG-{subj}_{pp}{sess}_ica-raw.fif"),
            #                  preload=True)
            # all_chans = raw.ch_names
            # raw.set_eeg_reference(projection=True) # average reference
            # epo = mne.make_fixed_length_epochs(raw, .5)
            epo = mne.read_epochs(join(sess_dir, f"MT-YG-{subj}_{pp}{sess}-epo.fif"),
                                  preload=True)
            #epo.crop(tmin=-.25, tmax=.25)
            epo.pick_types(eeg=True)
            # fmin, fmax = 4., 6.5
            # sfreq = epo.info['sfreq']  # the sampling frequency
            # tmin = epo.times[0]  # exclude the baseline period
            # con = spectral_connectivity_epochs(epo, method='wpli',
            #                                    mode='cwt_morlet', sfreq=sfreq,
            #                                    cwt_freqs=np.array([4., 4.5, 5.,
            #                                                        5.5, 6.,
            #                                                        6.5]),
            #                                    cwt_n_cycles=2,
            #                                    fmin=fmin, fmax=fmax,
            #                                    faverage=True,
            #                                    tmin=tmin, n_jobs=8)
            # c = con.get_data(output="dense")[:, :, 0]
            # c = c.max(axis=-1)
            dpte = epo_dPTE(epo.get_data(), freqs, epo.info["sfreq"], n_jobs=8,
                            n_cycles=2)
            c = dpte.mean(axis=0)
            plot_sensors_connectivity(epo.info, c)
