from os.path import join
from os import listdir
import numpy as np
import mne
import re
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                            apply_inverse_epochs)
from mne.inverse_sparse.subspace_pursuit import (subspace_pursuit,
                                                 subspace_pursuit_level,
                                                 mix_patch_forwards)
import matplotlib.pyplot as plt
import pickle
from utils import make_brain_image, locate_vertex, fill_dpte_mat
import pandas as pd
import seaborn as sns
from mne_connectivity import spectral_connectivity_epochs as sce
from dPTE import epo_dPTE
from itertools import product
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

stim_key = {"MT-YG-120":{"Session1":"active", "Session2":"sham"},
            "MT-YG-121":{"Session1":"active", "Session2":"sham"},
            "MT-YG-128":{"Session1":"sham", "Session2":"active"},
            "MT-YG-127":{"Session1":"sham", "Session2":"active"},
            "MT-YG-138":{"Session1":"sham", "Session2":"active"},
            "MT-YG-133":{"Session1":"active", "Session2":"sham"},
            "MT-YG-136":{"Session1":"active", "Session2":"sham"},
            "MT-YG-132":{"Session1":"sham", "Session2":"active"},
            "MT-YG-134":{"Session1":"active", "Session2":"sham"},
            "MT-YG-125":{"Session1":"sham", "Session2":"active"},
            "MT-YG-131":{"Session1":"sham", "Session2":"active"},
            "MT-YG-122":{"Session1":"active", "Session2":"sham"},
            "MT-YG-124":{"Session1":"sham", "Session2":"active"},
            "MT-YG-139":{"Session1":"active", "Session2":"sham"},
            "MT-YG-141":{"Session1":"active", "Session2":"sham"},
            "MT-YG-140":{"Session1":"sham", "Session2":"active"},
            "MT-YG-148":{"Session1":"sham", "Session2":"active"},
            "MT-YG-144":{"Session1":"sham", "Session2":"active"},
            "MT-YG-146":{"Session1":"active", "Session2":"sham"},
            "MT-YG-142":{"Session1":"active", "Session2":"sham"},
            "MT-YG-137":{"Session1":"active", "Session2":"sham"},
            "MT-YG-147":{"Session1":"sham", "Session2":"active"}
            }


subjs = listdir(data_dir)
preposts = ["pre", "post"]
exclu = ["MT-YG-124"]

snr = 1.
lambda2 = 1. / snr ** 2

bursts = True

signal_dict = {"subj":[], "stim":[], "pp":[], "reg":[], "time":[], "amp":[]}
amp_dict = {"subj":[], "stim":[], "pp":[], "reg":[], "amp":[]}
cnx_dict = {"subj":[], "stim":[], "pp":[], "from_reg":[], "to_reg":[],
            "wpli":[], "dpli":[], "dpte":[]}

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
        ctxfwd_file = f"{subj}_{sess}_restr-fwd.fif"
        ctx_fwd = mne.read_forward_solution(join(sess_dir, ctxfwd_file))
        subfwd_file = f"{subj}_{sess}_sub_restr-fwd.fif"
        sub_fwd = mne.read_forward_solution(join(sess_dir, subfwd_file))

        pp_epos = {}
        pp_inds = {}
        pp_epo_idx = 0
        all_bads = []

        for pp_idx, pp in enumerate(preposts):
            epo = mne.read_epochs(join(sess_dir,
                                       f"{subj}_{sess}_{pp}-epo.fif"),
                                  preload=True)
            if bursts:
                epo.crop(tmin=-.4, tmax=.4)
            pp_epos[pp] = epo
            pp_inds[pp] = [pp_epo_idx, pp_epo_idx + len(epo)]
            pp_epo_idx = len(epo)
            all_bads.extend(epo.info["bads"])

        for pp in preposts:
            pp_epos[pp].del_proj()
            pp_epos[pp].info["bads"] = all_bads
        epo = mne.concatenate_epochs(list(pp_epos.values()))

        epo.set_eeg_reference(projection=True)
        epo.apply_proj()
        if bursts:
            cov = mne.compute_covariance(epo, keep_sample_mean=False)
        else:
            cov = mne.make_ad_hoc_cov(epo.info)
        inst = epo

        mix_fwd = mix_patch_forwards(ctx_fwd, sub_fwd, inst.info, trans, bem)
        inv_op = make_inverse_operator(inst.info, mix_fwd, cov, loose=0,
                                       fixed=True, depth=None)
        if isinstance(inst, mne.BaseEpochs):
            stc = apply_inverse_epochs(inst, inv_op, lambda2, method="MNE")

        # assigns stc indices to regions
        verts = stc[0].vertices
        idx = 0
        reg_inds = {"middlefrontal-lh":[], "superiorfrontal-lh":[],
                    "temporal-lh":[], "hippocampus-lh":[],
                    "middlefrontal-rh":[], "superiorfrontal-rh":[],
                    "temporal-rh":[], "hippocampus-rh":[]}
        for vert_idx, vert in enumerate(verts):
            for vtx in vert:
                if vert_idx == 0:
                    reg_name = locate_vertex(vtx, lh_labels).name
                elif vert_idx == 1:
                    reg_name = locate_vertex(vtx, rh_labels).name
                else:
                    reg_name = ""
                # if "pars" in reg_name and vert_idx == 0:
                #     reg_inds["frontal-lh"].append(idx)
                # elif "pars" in reg_name and vert_idx == 1:
                #     reg_inds["frontal-rh"].append(idx)
                # elif "supra" in reg_name and vert_idx == 0:
                #     reg_inds["posterior-lh"].append(idx)
                # elif "supra" in reg_name and vert_idx == 1:
                #     reg_inds["posterior-rh"].append(idx)
                if "middlefrontal" in reg_name and vert_idx == 0:
                    reg_inds["middlefrontal-lh"].append(idx)
                elif "middlefrontal" in reg_name and vert_idx == 1:
                    reg_inds["middlefrontal-rh"].append(idx)
                elif "superiorfrontal" in reg_name and vert_idx == 0:
                    reg_inds["superiorfrontal-lh"].append(idx)
                elif "superiorfrontal" in reg_name and vert_idx == 1:
                    reg_inds["superiorfrontal-rh"].append(idx)
                elif ("temporal" in reg_name or "entorhinal" in reg_name) and vert_idx == 0:
                    reg_inds["temporal-lh"].append(idx)
                elif ("temporal" in reg_name  or "entorhinal" in reg_name) and vert_idx == 1:
                    reg_inds["temporal-rh"].append(idx)
                elif vert_idx == 2:
                    reg_inds["hippocampus-lh"].append(idx)
                elif vert_idx == 3:
                    reg_inds["hippocampus-rh"].append(idx)
                else:
                    raise ValueError("Can't identify region.")
                idx += 1
        # grab the strongest singal from each region
        data = np.array([s.data for s in stc])
        picked_inds = []
        for k, v in reg_inds.items():
            these_data = data[:, v].mean(axis=0)
            picked_inds.append(v[np.linalg.norm(these_data, axis=-1).argmax()])
        picked_data = data[:, picked_inds]
        # build up the database
        # for connectivity later
        sig_n = len(picked_inds)
        cnx_combos = product(np.arange(sig_n), np.arange(sig_n))
        cnx_combos = [cc for cc in cnx_combos if cc[0]!=cc[1]]
        cnx_combos = tuple(zip(*cnx_combos))
        # signals
        times = stc[0].times
        data_n = len(times)
        epo_n = len(these_data)
        for pp in preposts:
            these_data = picked_data[pp_inds[pp][0]:pp_inds[pp][1]]
            for k_idx, k in enumerate(reg_inds.keys()):
                for epo_idx in range(len(these_data)):
                    signal_dict["subj"].extend([subj] * data_n)
                    signal_dict["stim"].extend([stim_key[subj][sess]] * data_n)
                    signal_dict["pp"].extend([pp] * data_n)
                    signal_dict["reg"].extend([k] * data_n)
                    signal_dict["time"].extend(times)
                    signal_dict["amp"].extend(these_data[epo_idx, k_idx])
                amp_dict["subj"].append(subj)
                amp_dict["stim"].append(stim_key[subj][sess])
                amp_dict["pp"].append(pp)
                amp_dict["reg"].append(k)
                norm = np.linalg.norm(np.linalg.norm(these_data[:, k_idx],
                                      axis=-1), axis=0) / (epo_n + data_n)
                amp_dict["amp"].append(norm)

            # connectivity
            reg_names = list(reg_inds.keys())
            con = sce(these_data, method="wpli", fmin=4, fmax=7, faverage=True,
                      sfreq=epo.info["sfreq"], mt_bandwidth=3.5,
                      indices=cnx_combos)
            wpli_data = np.squeeze(con.get_data())
            con = sce(these_data, method="dpli", fmin=4, fmax=7, faverage=True,
                      sfreq=epo.info["sfreq"], mt_bandwidth=3.5,
                      indices=cnx_combos)
            dpli_data = np.squeeze(con.get_data())

            # dPTE
            if bursts:
                dpte = epo_dPTE(these_data, [4, 5, 6, 7, 8], epo.info["sfreq"],
                                n_cycles=[1, 2, 3, 3, 3])
            else:
                dpte = epo_dPTE(these_data, [4, 5, 6, 7], epo.info["sfreq"],
                                n_cycles=[3, 5, 7, 7])
            dpte = fill_dpte_mat(dpte.mean(axis=0))

            for cnx_idx, (w_cnx, d_cnx) in enumerate(zip(wpli_data, dpli_data)):
                cnx_dict["subj"].append(subj)
                cnx_dict["stim"].append(stim_key[subj][sess])
                cnx_dict["pp"].append(pp)
                cnx_dict["from_reg"].append(reg_names[cnx_combos[0][cnx_idx]])
                cnx_dict["to_reg"].append(reg_names[cnx_combos[1][cnx_idx]])
                cnx_dict["wpli"].append(w_cnx)
                cnx_dict["dpli"].append(d_cnx)
                cnx_dict["dpte"].append(dpte[cnx_combos[0][cnx_idx],
                                             cnx_combos[1][cnx_idx]])

cnx_df = pd.DataFrame.from_dict(cnx_dict)
cnx_df.to_pickle(join(fig_dir, "cnx.pickle"))
signal_df = pd.DataFrame.from_dict(signal_dict)
signal_df.to_pickle(join(fig_dir, "signal.pickle"))
amp_df = pd.DataFrame.from_dict(amp_dict)
amp_df.to_pickle(join(fig_dir, "amp.pickle"))

cnx_df.to_csv(join(fig_dir, "cnx.csv"))
amp_df.to_csv(join(fig_dir, "amp.csv"))
