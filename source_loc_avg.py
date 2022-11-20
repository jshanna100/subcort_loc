from os.path import join
from os import listdir
import numpy as np
import mne
import re
from mne.inverse_sparse.subspace_pursuit import (subspace_pursuit,
                                                 subspace_pursuit_level)
import matplotlib.pyplot as plt
import pickle
from utils import make_brain_image, locate_vertex
import pandas as pd
import seaborn as sns
plt.ion()

colors = [f"tab:{x}" for x in
          ["blue", "orange", "green", "red", "purple", "pink", "cyan", "olive"]]

views = {"left":{"view":"lateral", "distance":625, "hemi":"lh"},
         "right":{"view":"lateral", "distance":625, "hemi":"rh"},
         "upper":{"view":"dorsal", "distance":650,
                  "focalpoint":(-.77, 3.88, -21.53)},
         "caudal":{"view":"caudal", "distance":650}
        }

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
        for pp_idx, pp in enumerate(preposts):
            epo = mne.read_epochs(join(sess_dir,
                                       f"{subj}_{sess}_{pp}-epo.fif"),
                                  preload=True)["peak"]
            epo.crop(tmin=-.25, tmax=.25)
            cnx = {"method":"wpli", "fmin":4, "fmax":7, "sfreq":epo.info["sfreq"]}
            cov = mne.compute_covariance(epo, keep_sample_mean=False)

            # subspace pursuit - amplitude
            amp_out, fwds, resid = subspace_pursuit(subj, ["ico1", "ico2"], bem,
                                                   epo, cov, trans, [4, 4], lambda2,
                                                   fwd0=fwd0, return_as_dipoles=False,
                                                   subjects_dir=subjects_dir, mu=.5,
                                                   cnx=None, return_fwds=True, n_jobs=16,
                                                   patch_comp_n=2)
            # unpack and classify everything
            amp_out = list(amp_out)
            data = np.array([ss.data for ss in amp_out])
            verts = amp_out[0].vertices
            df_dict = {"Reg":[], "Amp":[], "Time":[]}
            for vtx_idx, vtx in enumerate(np.concatenate(verts)):
                reg_name = locate_vertex(vtx, labels).name
                for epo_idx in range(len(data)):
                    for x_idx in range(data.shape[-1]):
                        df_dict["Reg"].append(reg_name)
                        df_dict["Amp"].append(data[epo_idx, vtx_idx, x_idx])
                        df_dict["Time"].append(epo.times[x_idx])
            df = pd.DataFrame.from_dict(df_dict)
            sns.lineplot(data=df, x="Time", y="Amp", hue="Reg")
            breakpoint()

            brain = mne.viz.Brain(subj, hemi="both", surf="inflated")
            for hemi_idx, hemi in enumerate(["lh", "rh"]):
                brain.add_foci(verts[hemi_idx], coords_as_verts=True, hemi=hemi,
                               color="red", alpha=0.4)

            amp_img = make_brain_image(views, brain)
            plt.imshow(amp_img)
            breakpoint()


            #
            # #subcortical
            # mix_src = fwds[-1]["src"] + sub_fwd["src"]
            # mix_fwd = mne.make_forward_solution(epo.info, trans, mix_src, bem)
            # mix_fwd = mne.convert_forward_solution(mix_fwd, force_fixed=True)
            # mix_gain = np.hstack((fwds[-1]["sol"]["data"], sub_fwd["sol"]["data"]))
            # mix_fwd["sol"]["data"] = mix_gain
            # mix_stc, est_fwd, var_expl, resid = subspace_pursuit_level(mix_fwd, epo, cov,
            #                                                            3, .5, lambda2)
            # plt.figure()
            # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            # if isinstance(mix_stc, tuple):
            #     times = mix_stc[0].times
            #     data = np.array([ss.data for ss in mix_stc])
            #     plt.plot(times, data.mean(axis=0).T)
            #     for src_idx, col in zip(np.arange(data.shape[1]),
            #                             colors):
            #         plt.plot(times, data[:, src_idx, ].T, color=col, alpha=.1)
            #     verts = mix_stc[0].vertices
            # else:
            #     verts = mix_stc.vertices
            #     plt.plot(mix_stc.times, mix_stc.data.T)
            #
            # brain = mne.viz.Brain(subj, hemi="both", surf="inflated")
            # brain.add_foci(verts[0], coords_as_verts=True,
            #                hemi="lh", color="red")
            # brain.add_foci(verts[1], coords_as_verts=True,
            #                hemi="rh", color="red")
            # print(verts)
