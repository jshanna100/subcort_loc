import mne
import numpy as np
from dPTE import epo_dPTE
from os.path import join
from os import listdir
import re
from mne.beamformer import make_dics, apply_dics_epochs

root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
fig_dir = join(mem_dir, "figures")
data_dir = join(root_dir, mem_dir, "02_MemTask")
subjects_dir = root_dir + "hdd/freesurfer/subjects"

inv_method="MNE"
snr = 1.0
lambda2 = 1.0 / snr ** 2

preposts = ["pre", "post"]

n_jobs = 8

subjs = listdir(data_dir)
for subj in subjs:
    match = re.match("MT-YG-(\d{3})", subj)
    if not match:
        continue
    subj_dir = join(data_dir, subj)
    for sess in listdir(subj_dir):
        if not re.match("Session\d", sess):
            continue
        sess_dir = join(subj_dir, sess, "EEG")
        ctxfwd_file = f"{subj}_{sess}_ctx-fwd.fif"
        fwd = mne.read_forward_solution(join(sess_dir, ctxfwd_file))
        filter = make_dics()
        for pp_idx, pp in enumerate(preposts):
            epo = mne.read_epochs(join(sess_dir,
                                       f"{subj}_{sess}_{pp}-epo.fif"),
                                  preload=True)
            epo.set_eeg_reference(projection=True)
            cov = mne.make_ad_hoc_cov(epo.info)
