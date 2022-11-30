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
overwrite = True

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
            outfile = f"{subj}_{sess}_{pp}-epo.fif"
            if outfile in listdir(sess_dir) and not overwrite:
                print("Already exists. Skipping...")
                continue
            raw = mne.io.Raw(join(sess_dir, f"{subj}_{sess}_{pp}_ica-raw.fif"),
                             preload=True)
            raw.filter(l_freq=4, h_freq=8, picks=raw.ch_names) # theta band
            events = mne.make_fixed_length_events(raw, 2)
            epo = mne.Epochs(raw, events, tmin=0, tmax=2, baseline=None,
                             reject={"eeg":100e-6})
            epo.save(join(sess_dir, outfile),
                     overwrite=True)
