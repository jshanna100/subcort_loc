from os.path import join
from os import listdir
import numpy as np
import mne
import re
import matplotlib.pyplot as plt
import pickle
from utils import make_brain_image, locate_vertex
import pandas as pd
import seaborn as sns
from principle_angles import fwd_sa
from mne.inverse_sparse.subspace_pursuit import mix_patch_forwards

plt.ion()

sc_base = ["Caudate", "Putamen", "Hippocampus", "Amygdala"]
sc_names = [f"Left-{x}" for x in sc_base] +  [f"Right-{x}" for x in sc_base]

colors = [f"tab:{x}" for x in
          ["blue", "orange", "green", "red", "purple", "pink", "cyan", "olive"]]

views = {"left":{"view":"lateral", "distance":625, "hemi":"lh"},
         "right":{"view":"lateral", "distance":625, "hemi":"rh"},
         "upper":{"view":"dorsal", "distance":550,
                  "focalpoint":(-.77, 3.88, -21.53)},
         "caudal":{"view":"caudal", "distance":550}
        }

root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
fig_dir = join(mem_dir, "figures")
data_dir = join(root_dir, mem_dir, "02_MemTask")
subjects_dir = root_dir + "hdd/freesurfer/subjects"

subjs = listdir(data_dir)

ctx_ctx, ctx_sub, all_stc = [], [], []
for subj in subjs:
    match = re.match("MT-YG-(\d{3})", subj)
    if not match:
        continue
    subj_dir = join(data_dir, subj)
    try:
        bem = mne.read_bem_solution(join(subj_dir, f"{subj}-bem.fif"))
    except:
        print("BEM not found. Skipping...")
        continue
    labels = mne.read_labels_from_annot(subj, "aparc",
                                        subjects_dir=subjects_dir)
    lh_labels = [lab for lab in labels if "lh" in lab.name]
    rh_labels = [lab for lab in labels if "rh" in lab.name]
    for sess in listdir(subj_dir):
        if not re.match("Session\d", sess):
            continue
        sess_dir = join(subj_dir, sess, "EEG")
        trans = join(sess_dir, f"{subj}_{sess}_auto-trans.fif")
        ctxfwd_file = f"{subj}_{sess}_restr_roi-fwd.fif"
        ctx_fwd = mne.read_forward_solution(join(sess_dir, ctxfwd_file))
        # wholectx_fwd_file = f"{subj}_{sess}_ctx-fwd.fif"
        # wholectx_fwd = mne.read_forward_solution(join(sess_dir,
        #                                               wholectx_fwd_file))
        # wholectx_fwd = mne.convert_forward_solution(wholectx_fwd,
        #                                             force_fixed=True)
        subfwd_file = f"{subj}_{sess}_sub_restr-fwd.fif"
        sub_fwd = mne.read_forward_solution(join(sess_dir, subfwd_file))

        epo = mne.read_epochs(join(sess_dir,
                                   f"{subj}_{sess}_pre-epo.fif"),
                              preload=True)

        epo.set_eeg_reference(projection=True)
        epo.apply_proj()
        inst = epo
        mix_fwd = mix_patch_forwards(ctx_fwd, sub_fwd, inst.info, trans, bem)

        ctx_n = sum(len(s["vertno"]) for s in mix_fwd["src"][:2])
        sub_n = sum(len(s["vertno"]) for s in mix_fwd["src"][2:])
        ctx_gain = mix_fwd["sol"]["data"][:, :ctx_n]
        sub_gain = mix_fwd["sol"]["data"][:, ctx_n:]
        ctx_ctx.append(fwd_sa(ctx_gain, ctx_gain))
        ctx_sub.append(fwd_sa(sub_gain, ctx_gain))
