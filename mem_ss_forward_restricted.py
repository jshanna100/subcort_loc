import mne
from os.path import join
from mne.inverse_sparse.subspace_pursuit import make_patch_forward
from utils import make_restricted_forward
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

label_groups = [["G_front_sup-lh"],
                ["G_front_middle-lh"],
                ["G_temporal_middle-lh"],
                ["G_front_sup-rh"],
                ["G_front_middle-rh"],
                ["G_temporal_middle-rh"]]

sc_base = ["Hippocampus"]
sc_names = [f"Left-{x}" for x in sc_base] +  [f"Right-{x}" for x in sc_base]

n_jobs = 16
subjects_dir = root_dir + "hdd/freesurfer/subjects"
overwrite = True

subjs = listdir(data_dir)

parc = "aparc.a2009s"

for subj in subjs:
    if "MT-YG" not in subj:
        continue
    subj_dir = join(data_dir, subj)
    for sess in listdir(subj_dir):
        if not re.match("Session\d", sess):
            continue
        sess_dir = join(subj_dir, sess, "EEG")
        trans = join(sess_dir, f"{subj}_{sess}_auto-trans.fif")
        if f"{subj}_{sess}_restr-fwd.fif" in listdir(sess_dir) and not overwrite:
            print("Forward model already exists. Skipping...")
            continue
        if f"{subj}_{sess}_auto-trans.fif" not in listdir(sess_dir):
            print("No trans file found. Skipping...")
            continue

        # set up labels for restricted cortical space
        labels = mne.read_labels_from_annot(subj, parc,
                                            subjects_dir=subjects_dir)
        comb_labels = []
        for lg in label_groups:
            labs = [lab for lab in labels if lab.name in lg]
            lab = labs[0]
            for l in labs[1:]:
                lab += l
            comb_labels.append(lab)

        ctx_src = mne.read_source_spaces(join(subj_dir,
                                              f"{subj}-src.fif"))
        bem = mne.read_bem_solution(join(subj_dir, f"{subj}-bem.fif"))
        raw = mne.io.Raw(join(sess_dir, f"{subj}_{sess}_pre_ica-raw.fif"))

        ctx_fwd = make_restricted_forward(subj, comb_labels, bem, raw.info,
                                          trans, subjects_dir=subjects_dir,
                                          n_jobs=16)
        mne.write_forward_solution(join(sess_dir,
                                        f"{subj}_{sess}_restr-fwd.fif"),
                                   ctx_fwd, overwrite=True)

        sub_fwd = make_patch_forward(subj, None, bem, raw.info, trans,
                                     volume=True, volume_label=sc_names,
                                     subjects_dir=subjects_dir, n_jobs=16)
        mne.write_forward_solution(join(sess_dir,
                                        f"{subj}_{sess}_sub_restr-fwd.fif"),
                                   sub_fwd, overwrite=True)

        # make with one source per ROI/patch for doing PA calculations later
        # set up labels for restricted cortical space

        # ctx_fwd = make_restricted_forward(subj, comb_labels, bem, raw.info,
        #                                   trans, subjects_dir=subjects_dir,
        #                                   n_jobs=16, patch_comp_n=1)
        # mne.write_forward_solution(join(sess_dir,
        #                                 f"{subj}_{sess}_restr_1eig-fwd.fif"),
        #                            ctx_fwd, overwrite=True)
        # sub_fwd = make_patch_forward(subj, None, bem, raw.info, trans,
        #                              volume=True, volume_label=sc_names,
        #                              subjects_dir=subjects_dir, n_jobs=16,
        #                              patch_comp_n=1)
        # mne.write_forward_solution(join(sess_dir,
        #                                 f"{subj}_{sess}_sub_restr_1eig-fwd.fif"),
        #                            sub_fwd, overwrite=True)
