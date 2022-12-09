import mne
import numpy as np
from dPTE import epo_dPTE
from os.path import join
from os import listdir
import re
from mne_connectivity import spectral_connectivity_epochs as sce
from cnx_utils import TriuSparse

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

fs_labels = mne.read_labels_from_annot("fsaverage", "RegionGrowing_70",
                                       subjects_dir=subjects_dir)
subjs = listdir(data_dir)
#subjs = ['MT-YG-135', 'MT-YG-142', 'MT-YG-148', 'MT-YG-147', 'MT-YG-144']

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
        labels = mne.morph_labels(fs_labels, subj,
                                  subject_from="fsaverage",
                                  subjects_dir=subjects_dir)
        for pp_idx, pp in enumerate(preposts):
            try:
                epo = mne.read_epochs(join(sess_dir,
                                           f"{subj}_{sess}_{pp}-epo.fif"),
                                      preload=True)
            except:
                continue
            epo.set_eeg_reference(projection=True)
            cov = mne.make_ad_hoc_cov(epo.info)
            inv_op = mne.minimum_norm.make_inverse_operator(epo.info, fwd, cov)
            stcs = mne.minimum_norm.apply_inverse_epochs(epo, inv_op, lambda2,
                                                        method=inv_method,
                                                        pick_ori="normal")
            samp_n = stcs[0].data.shape[1]
            l_arr = [np.linalg.norm(stc.data, axis=1, keepdims=True) / samp_n for stc in stcs]
            l_arr = np.array(l_arr).mean(axis=0)
            new_stc = stcs[0].copy()
            new_stc.data = l_arr
            morph = mne.compute_source_morph(new_stc, subject_from=subj,
                                             subject_to="fsaverage")
            new_stc = morph.apply(new_stc)
            new_stc.save(join(sess_dir, f"{subj}_{sess}_{pp}"), overwrite=True)
            del l_arr, new_stc
