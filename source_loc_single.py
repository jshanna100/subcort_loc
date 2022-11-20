from os.path import join
from os import listdir
import numpy as np
import mne
import re
from mne.inverse_sparse.subspace_pursuit import (subspace_pursuit,
                                                 subspace_pursuit_level)
import matplotlib.pyplot as plt
import pickle
#plt.ion()

def cort_histos(verts, labels, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    lab_n = len(labels)
    lab_names = [lab.name for lab in labels]
    # count up how many verts land in each label
    label_hits = [0 for x in range(lab_n)]
    for lab_idx, lab in enumerate(labels):
        inter = np.intersect1d(verts, lab.vertices)
        label_hits[lab_idx] += len(inter)
    ax.bar(np.arange(lab_n), label_hits)
    ax.set_xticks(np.arange(lab_n), labels=lab_names, rotation=90, weight="bold")

    output = {lab_name:lab_hit for lab_name, lab_hit in zip(lab_names, label_hits)}

    return output


root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
fig_dir = join(mem_dir, "figures")
data_dir = join(root_dir, mem_dir, "02_MemTask")
subjects_dir = root_dir + "hdd/freesurfer/subjects"

subjs = listdir(data_dir)
preposts = ["pre", "post"]

try:
    with open(join(fig_dir, "all_hits.pickle"), "rb") as f:
        all_hits = pickle.load(f)
except:
    all_hits = []

snr = 1.
lambda2 = 1. / snr ** 2
cnx = None
for subj in subjs:
    match = re.match("MT-YG-(\d{3})", subj)
    if not match:
        continue
    subj_dir = join(data_dir, subj)
    try:
        bem = mne.read_bem_solution(join(subj_dir, f"{subj}-bem.fif"))
    except:
        continue
    src = mne.read_source_spaces(join(subj_dir, f"{subj}-src.fif"))
    labels = mne.read_labels_from_annot(subj, "aparc",
                                         subjects_dir=subjects_dir)
    lh_labels = [lab for lab in labels if "lh" in lab.name]
    rh_labels = [lab for lab in labels if "rh" in lab.name]
    # order by back to front
    lh_ypos = [lab.pos.mean(axis=0)[1] for lab in lh_labels]
    lh_labels = [lh_labels[x] for x in np.argsort(lh_ypos)]
    rh_ypos = [lab.pos.mean(axis=0)[1] for lab in rh_labels]
    rh_labels = [rh_labels[x] for x in np.argsort(rh_ypos)]
    for sess in listdir(subj_dir):
        if not re.match("Session\d", sess):
            continue
        if f"{subj}_{sess}.png" in listdir(fig_dir):
            continue
        sess_dir = join(subj_dir, sess, "EEG")
        trans = join(sess_dir, f"{subj}_session{sess[-1]}-trans.fif")
        fwd0_file = f"{subj}_{sess}_p0-fwd.fif"
        try:
            fwd0 = mne.read_forward_solution(join(sess_dir, fwd0_file))
        except:
            continue
        subfwd_file = f"{subj}_{sess}_sub_p0-fwd.fif"
        sub_fwd = mne.read_forward_solution(join(sess_dir, subfwd_file))
        fig, axes = plt.subplots(1, 2, figsize=(38.4, 21.6))
        pp_hits = {}
        for pp_idx, pp in enumerate(preposts):
            epo = mne.read_epochs(join(sess_dir,
                                       f"{subj}_{sess}_{pp}-epo.fif"),
                                  preload=True)["peak"]
            epo.crop(tmin=-.25, tmax=.25)
            #cnx = {"method":"wpli", "fmin":4, "fmax":7, "sfreq":epo.info["sfreq"]}
            cov = mne.compute_covariance(epo, keep_sample_mean=False)

            # one level ss by epoch
            verts = [[], []]
            for e_idx in range(len(epo)):
                stc, _, _, _ = subspace_pursuit_level(fwd0, epo[e_idx], cov, 4,
                                                      0.3, lambda2, cnx=None)
                verts[0].extend(stc[0].vertices[0])
                verts[1].extend(stc[0].vertices[1])
            hits = cort_histos(np.concatenate(verts), lh_labels + rh_labels,
                               ax=axes[pp_idx])
            pp_hits[pp] = hits
            axes[pp_idx].set_title(pp, fontsize=28, weight="bold")

        all_hits.append(pp_hits)
        new_max = np.max((np.max(list(pp_hits["pre"].values())),
                         np.max(list(pp_hits["post"].values()))))
        for ax in axes:
            ax.set_ylim(0, new_max)
        plt.suptitle(f"{subj} {sess}", fontsize=28, weight="bold")
        plt.tight_layout()
        fig.savefig(join(fig_dir, f"{subj}_{sess}.png"))
        plt.close()

with open(join(fig_dir, "all_hits.pickle"), "wb") as f:
    pickle.dump(all_hits, f)

# total up all the hits and plot
all_labels = [lab.name for lab in lh_labels + rh_labels]
lab_n = len(all_labels)
fig, axes = plt.subplots(1, 2, figsize=(38.4, 21.6))
maxes = []
for pp, ax in zip(["pre", "post"], axes):
    label_hits = [0 for x in range(lab_n)]
    for lab_idx, lab in enumerate(all_labels):
        for ah in all_hits:
            label_hits[lab_idx] += ah[pp][lab]
    maxes.append(np.max(label_hits))
    ax.bar(np.arange(lab_n), label_hits)
    ax.set_xticks(np.arange(lab_n), labels=all_labels, rotation=90, weight="bold")
    ax.set_title(pp, fontsize=28, weight="bold")
for ax in axes:
    ax.set_ylim(0, np.max(maxes))
plt.suptitle(f"grand", fontsize=28, weight="bold")
plt.tight_layout()
fig.savefig(join(fig_dir, "grand.png"))
