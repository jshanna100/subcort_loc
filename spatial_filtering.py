import mne
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import re
from mne.minimum_norm import make_inverse_operator
from mne.minimum_norm.resolution_matrix import _get_matrix_from_inverse_operator
from mne.utils.numerics import _PCA
plt.ion()

"""
Create spatial filters which maximise signal from given brain regions
"""

def disp_mat_comps(mat, info, n_components=None, title=None):
    pca = _PCA(n_components=n_components)
    u, s, v = pca._fit(mat.T)
    comps = u[:, :pca.n_components_] * s[:pca.n_components_]
    fig, axes = plt.subplots(1, pca.n_components_+1, squeeze=False)
    for idx, ax in enumerate(axes[0][:-1]):
        mne.viz.plot_topomap(comps.T[idx,], info, axes=ax)
    mne.viz.plot_topomap(mat.mean(axis=0), info, axes=axes[0][-1])
    plt.suptitle(title)
    plt.tight_layout()


def inv_mat_subset(inv, fwd, label):
    # get the inversion matrix and return a subset of it corresponding to
    # a label
    inv_mat = _get_matrix_from_inverse_operator(inv, fwd,
                                                method="MNE")
    # get inds in matrix that correspond to label vertices
    verts = np.concatenate([s["vertno"] for s in fwd["src"]])
    int_verts = np.intersect1d(verts, label.vertices)
    inds = np.array([np.where(verts==v)[0][0] for v in int_verts])
    return inv_mat[inds]


root_dir = "/home/jev/"
subjects_dir = join(root_dir, 'hdd', "freesurfer", "subjects")
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
data_dir = join(root_dir, mem_dir, "02_MemTask")

subjs = listdir(data_dir)
preposts = ["pre", "post"]

for subj in subjs:
    match = re.match("MT-YG-(\d{3})", subj)
    if not match:
        continue
    # get cortical labels
    ctx_labels = []
    ifg_labels = mne.read_labels_from_annot(subj, regexp='pars.*-lh',
                                            subjects_dir=subjects_dir)
    if len(ifg_labels) != 3:
        raise ValueError("Should be 3 IFG labels.")
    ifg_label = ifg_labels[0] + ifg_labels[1] + ifg_labels[2]
    ctx_labels.append(ifg_label)
    ctx_labels.append(mne.read_labels_from_annot(subj,
                                                 regexp='supramarginal-lh',
                                                 subjects_dir=subjects_dir)[0])
    subj_dir = join(data_dir, subj)
    for sess in listdir(subj_dir):
        if not re.match("Session\d", sess):
            continue
        sess_dir = join(subj_dir, sess, "EEG")
        try:
            fwd = mne.read_forward_solution(join(sess_dir,
                                                 f"{subj}_{sess}_ctx-fwd.fif"))
        except:
            continue
        for pp in preposts:
            epo = mne.read_epochs(join(sess_dir,
                                       f"MT-YG-{subj}_{sess}_{pp}-epo.fif"),
                                  preload=True)
            fwd = mne.convert_forward_solution(fwd, force_fixed=True)
            f_cov = mne.compute_covariance(epo["F_peak"], method="auto")
            p_cov = mne.compute_covariance(epo["P_peak"], method="auto")

            f_inv = make_inverse_operator(epo.info, fwd, f_cov, depth=0,
                                          loose=0,)
            f_inv_sub = inv_mat_subset(f_inv, fwd, ctx_labels[0])
            disp_mat_comps(f_inv_sub, epo.info, n_components=0.9, title="LIFG")

            p_inv = make_inverse_operator(epo.info, fwd, p_cov, depth=0,
                                          loose=0,)
            p_inv_sub = inv_mat_subset(p_inv, fwd, ctx_labels[1])
            disp_mat_comps(p_inv_sub, epo.info, n_components=0.9, title="LSM")
            breakpoint()
