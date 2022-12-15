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
exclu = ["MT-YG-124"]
n_jobs = 8

fs_labels = mne.read_labels_from_annot("fsaverage", "RegionGrowing_70",
                                       subjects_dir=subjects_dir)
subjs = listdir(data_dir)
subjs = ['MT-YG-128', 'MT-YG-134', 'MT-YG-125', 'MT-YG-137', 'MT-YG-138', 'MT-YG-127', 'MT-YG-132', 'MT-YG-139', 'MT-YG-120', 'MT-YG-133', 'MT-YG-140', 'MT-YG-121', 'MT-YG-135', 'MT-YG-142', 'MT-YG-148', 'MT-YG-147', 'MT-YG-144']
for subj in subjs:
    match = re.match("MT-YG-(\d{3})", subj)
    if not match:
        continue
    if subj in exclu:
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
            epo.crop(tmin=-.4, tmax=.4)
            #cov = mne.make_ad_hoc_cov(epo.info)
            cov = mne.compute_covariance(epo, keep_sample_mean=False)
            inv_op = mne.minimum_norm.make_inverse_operator(epo.info, fwd, cov)
            stcs = mne.minimum_norm.apply_inverse_epochs(epo, inv_op, lambda2,
                                                        method=inv_method,
                                                        pick_ori="normal")
            l_arr = [s.extract_label_time_course(labels, fwd["src"], mode="pca_flip").astype("float32") for s in stcs]
            l_arr = np.array(l_arr)

            # dPTE
            dPTE = epo_dPTE(l_arr, [4, 5, 6, 7, 8], epo.info["sfreq"],
                            n_cycles=[1, 2, 3, 3, 3])
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
