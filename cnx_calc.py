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
            epo = mne.read_epochs(join(sess_dir,
                                       f"{subj}_{sess}_{pp}-epo.fif"),
                                  preload=True)
            epo.set_eeg_reference(projection=True)
            cov = mne.make_ad_hoc_cov(epo.info)
            inv_op = mne.minimum_norm.make_inverse_operator(epo.info, fwd, cov)
            stcs = mne.minimum_norm.apply_inverse_epochs(epo, inv_op, lambda2,
                                                        method=inv_method,
                                                        pick_ori="normal")
            l_arr = [s.extract_label_time_course(labels, fwd["src"], mode="pca_flip").astype("float32") for s in stcs]
            l_arr = np.array(l_arr)

            # dPTE
            dPTE = epo_dPTE(l_arr, [4, 5, 6, 7, 8], epo.info["sfreq"],
                            n_cycles=[3, 5, 7, 7, 7], n_jobs=n_jobs)
            dPTE = TriuSparse(dPTE)
            dPTE.save(join(sess_dir, f"dPTE_{subj}_{sess}_{pp}.sps"))

            con = sce(l_arr, method="wpli", fmin=4, fmax=8, faverage=True,
                      sfreq=epo.info["sfreq"], mt_bandwidth=3.5)
            wpli = np.squeeze(con.get_data(output="dense")).T
            wpli = TriuSparse(wpli)
            wpli.save(join(sess_dir, f"wPLI_{subj}_{sess}_{pp}.sps"))

            con = sce(l_arr, method="dpli", fmin=4, fmax=8, faverage=True,
                      sfreq=epo.info["sfreq"], mt_bandwidth=3.5)
            dpli = np.squeeze(con.get_data(output="dense")).T
            dpli = TriuSparse(dpli)
            dpli.save(join(sess_dir, f"dPLI_{subj}_{sess}_{pp}.sps"))
