import mne
from os.path import join
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import zscore
import matplotlib.pyplot as plt
from os import listdir
import re
plt.ion()

"""
Identifies bursts of theta activity in frontal and parietal channels and
isolates them into epochs
"""

def phase_align(peaks, phase, win_ind):
    new_peaks = []
    win_half = win_ind//2
    for peak in peaks:
        new_ind = np.argmax(phase[peak-win_half:peak+win_half])
        new_ind += peak-win_half
        new_peaks.append(new_ind)
    return np.array(new_peaks)


root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
data_dir = join(root_dir, mem_dir, "02_MemTask")

subjs = listdir(data_dir)
preposts = ["pre", "post"]

for subj in subjs:
    match = re.match("MT-YG-(\d{3})", subj)
    if not match:
        continue
    subj_dir = join(data_dir, subj)
    for sess in listdir(subj_dir):
        if not re.match("Session\d", sess):
            continue
        sess_dir = join(subj_dir, sess, "EEG")
        for pp in preposts:
            raw = mne.io.Raw(join(sess_dir, f"{subj}_{sess}_{pp}_ica-raw.fif"),
                             preload=True)
            all_chans = raw.ch_names
            raw.filter(l_freq=4, h_freq=7, picks=raw.ch_names) # theta band
            # create frontal/parietal summary channels
            epo = mne.make_fixed_length_epochs(raw, 10)
            # calculate TFRs to see when theta power is strongest
            # we choose complex output so we can calculate both power and
            # instantaneous phase
            tfr = mne.time_frequency.tfr_morlet(epo, [4, 4.5, 5, 5.5, 6, 6.5, 7], 3,
                                                output="complex", return_itc=False,
                                                average=False)
            power = np.abs(tfr).mean(axis=2) * 1e+5
            power = zscore(power, axis=0)
            phase = np.angle(tfr).mean(axis=2)
            # tiresome work here to put the matrix back in raw format
            power = np.transpose(power, [0, 2, 1])
            power = power.reshape(power.shape[0] * power.shape[1], power.shape[2])
            power = np.transpose(power, [1, 0])
            phase = np.transpose(phase, [0, 2, 1])
            phase = phase.reshape(phase.shape[0] * phase.shape[1], phase.shape[2])
            phase = np.transpose(phase, [1, 0])


            # identify and annotate peaks in power
            prominence = 1.5
            win_ind = raw.time_as_index(.25)[0]
            peaks = find_peaks(power.mean(axis=0), prominence=prominence)[0]
            peaks = phase_align(peaks, phase[0,], win_ind)

            for p in peaks:
                raw.annotations.append(raw.times[p], 0, "peak")


            # finally epoch all theta events
            events = mne.events_from_annotations(raw, event_id={"peak":10})
            epo = mne.Epochs(raw, *events, tmin=-0.5, tmax=0.5, baseline=None,
                             event_repeated="merge")
            epo.save(join(sess_dir, f"{subj}_{sess}_{pp}-epo.fif"),
                     overwrite=True)
