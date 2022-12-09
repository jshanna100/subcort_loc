import mne
from os.path import join
from mne.inverse_sparse.subspace_pursuit import make_patch_forward
from os import listdir
import numpy as np
import re

'''
Build forward models
'''

root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
data_dir = join(root_dir, mem_dir, "02_MemTask")
proc_dir = join(root_dir, "hdd", "memtacs", "proc")
data_list = listdir(data_dir)
proc_list = listdir(proc_dir)

sc_base = ["Caudate", "Putamen", "Hippocampus", "Amygdala"]
sc_names = [f"Left-{x}" for x in sc_base] +  [f"Right-{x}" for x in sc_base]


n_jobs = 16
subjects_dir = root_dir + "hdd/freesurfer/subjects"
spacing = "ico5"
overwrite = True

subjs = listdir(data_dir)

for subj in subjs:
    subj_dir = join(data_dir, subj)
    for sess in listdir(subj_dir):
        if not re.match("Session\d", sess):
            continue
        sess_dir = join(subj_dir, sess, "EEG")
        trans = join(sess_dir, f"{subj}_{sess}_auto-trans.fif")
        if f"{subj}_{sess}_ctx-fwd.fif" in listdir(sess_dir) and not overwrite:
            print("Forward model already exists. Skipping...")
            continue
        if f"{subj}_{sess}_auto-trans.fif" not in listdir(sess_dir):
            continue
        ctx_src = mne.read_source_spaces(join(subj_dir,
                                              f"{subj}-src.fif"))
        bem = mne.read_bem_solution(join(subj_dir, f"{subj}-bem.fif"))
        raw = mne.io.Raw(join(sess_dir, f"{subj}_{sess}_pre_ica-raw.fif"))


        ctx_fwd = mne.make_forward_solution(raw.info, trans, ctx_src, bem,
                                            n_jobs=n_jobs)
        mne.write_forward_solution(join(sess_dir,
                                        f"{subj}_{sess}_ctx-fwd.fif"),
                                   ctx_fwd, overwrite=True)

        ico1_fwd = make_patch_forward(subj, "ico1", bem, raw.info,
                                      trans, subjects_dir=subjects_dir)
        mne.write_forward_solution(join(sess_dir,
                                        f"{subj}_{sess}_p0-fwd.fif"),
                                   ico1_fwd, overwrite=True)

        sub_fwd = make_patch_forward(subj, None, bem, raw.info, trans,
                                     volume=True, volume_label=sc_names,
                                     subjects_dir=subjects_dir, n_jobs=16)
        mne.write_forward_solution(join(sess_dir,
                                        f"{subj}_{sess}_sub_p0-fwd.fif"),
                                   sub_fwd, overwrite=True)
