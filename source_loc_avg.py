from os.path import join
from os import listdir
import numpy as np
import mne
import re
from mne.inverse_sparse.subspace_pursuit import (subspace_pursuit,
                                                 subspace_pursuit_level,
                                                 mix_patch_forwards)
import matplotlib.pyplot as plt
import pickle
from utils import make_brain_image, locate_vertex
import pandas as pd
import seaborn as sns
from dPTE import epo_dPTE
import matplotlib.pyplot as plt
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
data_dir = join(mem_dir, "02_MemTask")
subjects_dir = root_dir + "hdd/freesurfer/subjects"

n_jobs = 8
mode = "cwt_morlet"
mode = "multitaper"
cwt_freqs = np.array([4, 5, 6, 7, 8])
cwt_n_cycles = np.array([3, 5, 5, 5, 7])
mt_bandwidth = 3.5
s = 4
sub_s = 6
do_cnx = False
if do_cnx:
    s = int(s/2)
    sub_s = int(sub_s/2)

subjs = listdir(data_dir)
preposts = ["pre", "post"]
pp_axes = {"pre":["A", "B", "C", "D"], "post":["E", "F", "G", "H"]}

mos_str = """
          ABCD
          EFGH
          """

snr = 1.
lambda2 = 1. / snr ** 2

exclu = ["MT-YG-124"]
hit_df_dict = {"subj":[], "session":[], "reg":[]}
drop_df_dict = {"subj":[], "session":[], "reg":[]}

parc = "aparc.a2009s"
orig_labels = mne.read_labels_from_annot("fsaverage", parc,
                                         subjects_dir=subjects_dir)
for subj in subjs:
    match = re.match("MT-YG-(\d{3})", subj)
    if not match:
        continue
    if subj in exclu:
        continue
    subj_dir = join(data_dir, subj)
    try:
        bem = mne.read_bem_solution(join(subj_dir, f"{subj}-bem.fif"))
    except:
        continue
    src = mne.read_source_spaces(join(subj_dir, f"{subj}-src.fif"))
    labels = mne.morph_labels(orig_labels, subj, "fsaverage",
                              subjects_dir=subjects_dir)
    lh_labels = [lab for lab in labels if "lh" in lab.name]
    rh_labels = [lab for lab in labels if "rh" in lab.name]
    for sess in listdir(subj_dir):
        if not re.match("Session\d", sess):
            continue
        sess_dir = join(subj_dir, sess, "EEG")
        trans = join(sess_dir, f"{subj}_{sess}_auto-trans.fif")
        fwd0_file = f"{subj}_{sess}_p0-fwd.fif"
        try:
            fwd0 = mne.read_forward_solution(join(sess_dir, fwd0_file))
        except:
            continue
        subfwd_file = f"{subj}_{sess}_sub_p0-fwd.fif"
        sub_fwd = mne.read_forward_solution(join(sess_dir, subfwd_file))
        pp_hits = {}
        con_mats = []
        pp_reg_names = []
        fig, axes = plt.subplot_mosaic(mos_str, figsize=(38.4, 21.6))
        pp_epos = {}
        pp_inds = {}
        pp_epo_idx = 0
        all_bads = []
        for pp_idx, pp in enumerate(preposts):
            epo = mne.read_epochs(join(sess_dir,
                                       f"{subj}_{sess}_{pp}-epo.fif"),
                                  preload=True)
            epo.crop(tmin=-.4, tmax=.4)
            pp_epos[pp] = epo
            pp_inds[pp] = [pp_epo_idx, pp_epo_idx + len(epo)]
            pp_epo_idx = len(epo)
            all_bads.extend(epo.info["bads"])

        for pp in preposts:
            pp_epos[pp].del_proj()
            pp_epos[pp].info["bads"] = all_bads
            pp_epos[pp].set_eeg_reference(projection=True)
            pp_epos[pp].average().apply_proj().plot(axes=axes[pp_axes[pp][0]])
        epo = mne.concatenate_epochs(list(pp_epos.values()))
        if do_cnx:
            cnx = {"method":"wpli", "fmin":4, "fmax":8,
                   "sfreq":epo.info["sfreq"]}
            cnx = {"freqs":[4, 5, 6, 7, 8], "cycles":[1, 2, 3, 3, 3],
                   "sfreq":epo.info["sfreq"], "n_jobs":n_jobs}
        else:
            cnx = None
        cov = mne.compute_covariance(epo, keep_sample_mean=False)
        #cov = mne.make_ad_hoc_cov(epo.info)

        pp_evos = [ppe.average() for ppe in pp_epos.values()]

        # subspace pursuit - amplitude
        fwd0 = None
        amp_ctx, fwds, resid = subspace_pursuit(subj, ["ico1", "ico2"], bem,
                                                pp_evos, cov, trans, [s, s], lambda2,
                                                fwd0=fwd0, return_as_dipoles=False,
                                                subjects_dir=subjects_dir, mu=.5,
                                                cnx=cnx, return_fwds=True, n_jobs=n_jobs,
                                                patch_comp_n=2)

        # subcortical
        mix_fwd = mix_patch_forwards(fwds[-1], sub_fwd, epo.info, trans, bem)
        amp_mix, est_fwd, var_expl, resid = subspace_pursuit_level(mix_fwd, pp_evos, cov,
                                                                   sub_s, .5, lambda2,
                                                                   cnx=cnx)

        for pp_idx, pp in enumerate(preposts):
            data = amp_mix[pp_idx].data.copy()
            time_n = data.shape[-1]
            verts = amp_mix[pp_idx].vertices
            df_dict = {"Reg":[], "Amp":[], "Time":[]}
            reg_names = []
            data_idx = 0
            hemi_labels = [lh_labels, rh_labels]
            for src_idx, vert in enumerate(verts):
                for vtx in vert:
                    if src_idx < 2: # cortex
                        lab = locate_vertex(vtx, hemi_labels[src_idx])
                        if lab is None:
                            hemi_str = "lh" if src_idx == 0 else "rh"
                            reg_name = f"unknown-{hemi_str}"
                        else:
                            reg_name = lab.name
                    else: # subcortical
                        reg_name = sc_names[src_idx-2]
                    while reg_name in reg_names:
                        reg_name += "*"
                    reg_names.append(reg_name)
                    df_dict["Reg"].extend([reg_name] * time_n)
                    df_dict["Amp"].extend(list(data[data_idx,]))
                    df_dict["Time"].extend(epo.times)
                    data_idx += 1
            pp_reg_names.append(reg_names)

            # while we're here, note which ctx vertices dropped
            drop_names = []
            lh_drop = np.setdiff1d(amp_ctx[0].vertices[0],
                                   amp_mix[0].vertices[0])
            rh_drop = np.setdiff1d(amp_ctx[0].vertices[1],
                                   amp_mix[0].vertices[1])
            for src_idx, vert in enumerate([lh_drop, rh_drop]):
                lab = locate_vertex(vtx, hemi_labels[src_idx])
                if lab is None:
                    hemi_str = "lh" if src_idx == 0 else "rh"
                    reg_name = f"unknown-{hemi_str}"
                else:
                    reg_name = lab.name
                drop_names.append(reg_name)

            df = pd.DataFrame.from_dict(df_dict)
            sns.lineplot(data=df, x="Time", y="Amp", hue="Reg",
                         ax=axes[pp_axes[pp][2]])
            axes[pp_axes[pp][2]].set_title("Source amplitudes")

            brain = mne.viz.Brain(subj, hemi="both", surf="inflated")
            col_idx = 0
            for hemi_idx, hemi in enumerate(["lh", "rh"]):
                for vtx in verts[hemi_idx]:
                    brain.add_foci(vtx, coords_as_verts=True, scale_factor=3,
                                   hemi=hemi, color=colors[col_idx], alpha=0.7)
                    col_idx += 1

            amp_img = make_brain_image(views, brain, orient="square")
            brain.close()
            axes[pp_axes[pp][1]].imshow(amp_img)
            axes[pp_axes[pp][1]].axis("off")
            axes[pp_axes[pp][1]].set_title("Cortical sources")

            dpte = epo_dPTE(data[None,], [4, 5, 6, 7, 8], epo.info["sfreq"],
                            n_cycles=[1, 1, 2, 3, 3], n_jobs=n_jobs)
            dpte = dpte.mean(axis=0)
            # fill out the matrix
            dpte_inv = dpte.copy().T
            dpte_inv[dpte_inv>0] = 0.5 - (dpte_inv[dpte_inv>0] - 0.5)
            dpte += dpte_inv
            np.fill_diagonal(dpte, np.nan)
            con_mats.append(dpte)

        con_max = np.max([cm.max() for cm in con_mats])
        reg_n = len(reg_names)
        for pp_idx, pp in enumerate(preposts):
            sns.heatmap(con_mats[pp_idx], cmap="seismic", vmin=0.4, vmax=.6,
                        ax=axes[pp_axes[pp][3]])
            axes[pp_axes[pp][3]].set_xticks(np.arange(reg_n) + .5,
                                 labels=pp_reg_names[pp_idx], rotation=90, weight="bold")
            axes[pp_axes[pp][3]].set_yticks(np.arange(reg_n) + .5,
                                 labels=pp_reg_names[pp_idx], rotation=0, weight="bold")
            axes[pp_axes[pp][3]].set_title("dPTE")

        # build up a list of areas identified
        for rn in reg_names:
            hit_df_dict["subj"].append(subj)
            hit_df_dict["session"].append(sess)
            hit_df_dict["reg"].append(rn.replace("*", ""))
        for rn in drop_names:
            drop_df_dict["subj"].append(subj)
            drop_df_dict["session"].append(sess)
            drop_df_dict["reg"].append(rn)

        plt.suptitle(f"{subj} {sess}", fontsize=40)
        plt.annotate("Post", (0.19, 0.43), xycoords="figure fraction", fontsize=70)
        plt.annotate("Pre", (0.2, 0.91), xycoords="figure fraction", fontsize=70)
        plt.tight_layout()
        plt.savefig(join(fig_dir, f"{subj}_{sess}_src-cnx.png"))
        plt.close("all")


hit_df = pd.DataFrame.from_dict(hit_df_dict)
hit_df.to_pickle(join(fig_dir, f"hits_{parc}.pickle"))
drop_df = pd.DataFrame.from_dict(drop_df_dict)
drop_df.to_pickle(join(fig_dir, f"drops_{parc}.pickle"))

# plot cortical hits
regs = hit_df["reg"]
ctx_regs = [reg for reg in regs if "-lh" in reg or "-rh" in reg]
labels = mne.read_labels_from_annot("fsaverage", parc,
                                    subjects_dir=subjects_dir)
regs, counts = np.unique(regs, return_counts=True)
ctx_regs, ctx_counts = np.unique(ctx_regs, return_counts=True)
alphas = ((ctx_counts - ctx_counts.min()) /
          (ctx_counts.max() - ctx_counts.min())  * 0.8 + 0.2)
brain = mne.viz.Brain("fsaverage", hemi="both", surf="inflated")
for reg, alpha in zip(regs, alphas):
    if "-lh" not in reg and "-rh" not in reg:
        continue
    label = [lab for lab in labels if lab.name == reg]
    if len(label):
        label = label[0]
    else:
        continue
    brain.add_label(label, color="red", alpha=alpha, hemi=label.name[-2:])
img = make_brain_image(views, brain, orient="square")

fig, axes = plt.subplots(1, 2, figsize=(38.4, 21.6))
axes[0].imshow(img)
axes[0].axis("off")
axes[0].set_title("Cortical hits")
reg_order = [regs[idx] for idx in np.argsort(counts)]
sns.countplot(data=hit_df, x="reg", order=reg_order, ax=axes[1])
plt.xticks(rotation=90, weight="bold")
axes[1].set_title("All hits")
plt.tight_layout()
plt.savefig(join(fig_dir, f"hits_distro_{parc}.png"))

# plot cortical drops
regs = drop_df["reg"]
labels = mne.read_labels_from_annot("fsaverage", parc,
                                    subjects_dir=subjects_dir)
regs, counts = np.unique(regs, return_counts=True)
alphas = (counts - counts.min()) / (counts.max() - counts.min())  * 0.8 + 0.2
brain = mne.viz.Brain("fsaverage", hemi="both", surf="inflated")
for reg, alpha in zip(regs, alphas):
    if "-lh" not in reg and "-rh" not in reg:
        continue
    label = [lab for lab in labels if lab.name == reg]
    if len(label):
        label = label[0]
    else:
        continue
    brain.add_label(label, color="blue", alpha=alpha, hemi=label.name[-2:])
img = make_brain_image(views, brain, orient="square")

fig, axes = plt.subplots(1, 2, figsize=(38.4, 21.6))
axes[0].imshow(img)
axes[0].axis("off")
axes[0].set_title("Cortical drops")
reg_order = [regs[idx] for idx in np.argsort(counts)]
sns.countplot(data=drop_df, x="reg", order=reg_order, ax=axes[1])
plt.xticks(rotation=90, weight="bold")
axes[1].set_title("All drops")
plt.tight_layout()
plt.savefig(join(fig_dir, f"drop_distro_{parc}.png"))
