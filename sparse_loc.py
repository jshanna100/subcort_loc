import mne
from mne.inverse_sparse import mixed_norm, gamma_map
from mne.time_frequency import tfr_array_morlet
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from mne.forward.forward import _restrict_forward_to_src_sel
from itertools import combinations, product
from scipy.stats import circmean
from circular_hist import circ_hist_norm
plt.ion()

def principle_angle(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def plot_r(r_norms, r_means, quantiles=(0., .25, .50, .75, 1.)):
    fig = plt.figure()
    axes = [plt.subplot(221, projection="polar"),
            plt.subplot(222, projection="polar"),
            plt.subplot(223, projection="polar"),
            plt.subplot(224, projection="polar")]
    for ax_idx, quant_idx in enumerate(range(len(quantiles)-1)):
        r_inds = ((r_means > np.quantile(r_means, quantiles[quant_idx])) &
                  (r_means < np.quantile(r_means, quantiles[quant_idx+1])))
        these_rs = r_means[r_inds]
        circ_hist_norm(axes[ax_idx], these_rs)


def r_vector(rad):
    x_bar, y_bar = np.cos(rad).mean(), np.sin(rad).mean()
    r_mean = circmean(rad, low=-np.pi, high=np.pi)
    r_norm = np.linalg.norm((x_bar, y_bar))
    return r_mean, r_norm

def phase_diff(ph_a, ph_b):
    si = np.exp(1j * np.abs(ph_a - ph_b))
    return si

root_dir = "/home/jev/"
subjects_dir = join(root_dir, "freesurfer", "subjects")
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
data_dir = join(root_dir, mem_dir, "02_MemTask")

mni_roi_coords = {"LFrontal":np.array([-50, 32, 12]),
                  "LParietal":np.array([-42, -64, 36]),
                  "RFrontal":np.array([50, 32, 12]),
                  "RParietal":np.array([42, -64, 36])}
roi_cols = {"LFrontal":"red", "LParietal":"green", "RFrontal":"blue",
            "RParietal":"cyan"}

subjs = ["120"]
sessions = ["2"]
preposts = ["pre", "post"]
hemis = ["lh", "rh"]

for subj in subjs:
    subj_str = f"MT-YG-{subj}"
    subj_dir = join(data_dir, subj_str)
    for sess in sessions:
        for pp in preposts:
            sess_dir = join(subj_dir, f"Session{sess}", "EEG")
            epo = mne.read_epochs(join(sess_dir, f"MT-YG-{subj}_{pp}{sess}-epo.fif"),
                                  preload=True)
            epo.crop(tmin=-.4, tmax=.4)
            cov = mne.read_cov(join(sess_dir, f"MT-YG-{subj}_{pp}{sess}-cov.fif"))
            ctx_fwd = mne.read_forward_solution(join(sess_dir,
                                                f"MT-YG-{subj}_Session{sess}-ctx-fwd.fif"))
            mix_fwd = mne.read_forward_solution(join(sess_dir,
                                                f"MT-YG-{subj}_Session{sess}-mix-fwd.fif"))

            # principle angles
            # cortex to cortex
            gain = mix_fwd["sol"]["data"]
            gain = np.stack((gain[:, 0::3], gain[:, 1::3], gain[:, 2::3]))
            gain = np.linalg.norm(gain, axis=0)
            ctx_inds = mix_fwd["src"][0]["nuse"] + mix_fwd["src"][1]["nuse"]
            sub_inds = ctx_inds + sum([s["nuse"] for s in mix_fwd["src"][2:]])
            sub_n = sub_inds - ctx_inds
            rand_ctx_inds = np.random.randint(0, ctx_inds, sub_n)
            ctx_combs = list(combinations(rand_ctx_inds, 2))
            ctx_angles = np.zeros(len(ctx_combs))
            for idx, c_comb in enumerate(ctx_combs):
                ctx_angles[idx] = principle_angle(gain[:, c_comb[0]],
                                                  gain[:, c_comb[1]])
            plt.hist(ctx_angles, bins=100, alpha=0.3)
            # subcortical to subcortical
            sub_combs = list(combinations(np.arange(ctx_inds, sub_inds), 2))
            sub_angles = np.zeros(len(sub_combs))
            for idx, s_comb in enumerate(sub_combs):
                sub_angles[idx] = principle_angle(gain[:, s_comb[0]],
                                                  gain[:, s_comb[1]])
            plt.hist(sub_angles, bins=100, alpha=0.3)
            # subcortical to cortical
            sc_combs = list(product(rand_ctx_inds,
                                    np.arange(ctx_inds, sub_inds)))
            sc_combs = [sc_combs[i]
                        for i in np.random.randint(0, len(sc_combs),
                                                   len(sub_combs))]
            sc_angles = np.zeros(len(sc_combs))
            for idx, sc_comb in enumerate(sc_combs):
                sc_angles[idx] = principle_angle(gain[:, sc_comb[0]],
                                                 gain[:, sc_comb[1]])
            plt.hist(sc_angles, bins=100, alpha=0.3)

            sens = mne.sensitivity_map(ctx_fwd, ch_type="eeg")
            sens.data[sens.data<0.3] = 0
            lab = mne.stc_to_label(sens, smooth=False,
                                   subjects_dir=subjects_dir)
            ctx_fwd = mne.forward.restrict_forward_to_label(ctx_fwd, lab)

            evos = []
            for e_idx in range(len(epo)):
                evos.append(epo[e_idx].average())
            stcs = mixed_norm(evos, ctx_fwd, cov, alpha=50)

            # display constellation of sources
            brain = stcs[0].plot(subjects_dir=subjects_dir, hemi="both")
            verts = np.concatenate((stcs[0].lh_vertno, stcs[0].rh_vertno))
            mni_coords = []
            for v_idx, v in enumerate(verts):
                hemi = "lh" if v_idx < len(stcs[0].lh_vertno) else "rh"

                brain.add_foci(v, coords_as_verts=True, color="black", hemi=hemi,
                               scale_factor=0.5)
                mni_coords.append(mne.vertex_to_mni(v, hemis.index(hemi),
                                                    subj_str,
                                                    subjects_dir=subjects_dir))
            mni_coords = np.array(mni_coords)
            # identify foci nearest to MNI coords
            closest_inds = {}
            for k,v in mni_roi_coords.items():
                dists = np.sqrt(np.sum((mni_coords - v)**2, axis=1))
                closest_idx = np.argmin(dists)
                closest_inds[k] = closest_idx
                hemi = "lh" if closest_idx < len(stcs[0].lh_vertno) else "rh"
                brain.add_foci(verts[closest_idx], coords_as_verts=True,
                               color=roi_cols[k], hemi=hemi)

            # build sparse, mixed space
            # all_verts = np.concatenate([s["vertno"] for s in mix_fwd["src"]])
            # ctx_verts = np.concatenate([s["vertno"] for s in mix_fwd["src"][:2]])
            # sub_verts = np.concatenate([s["vertno"] for s in mix_fwd["src"][2:]])
            # ctx_sparse_verts_inds = [np.where(all_verts==v)[0][0]
            #                          for v in ctx_sparse_verts]
            # sub_verts_inds = [np.where(all_verts==v)[0][0] for v in sub_verts]

            vert_list = [s["vertno"] for s in mix_fwd["src"]]
            l_list = np.sort(np.array([closest_inds["LFrontal"],
                                      closest_inds["LParietal"]]))
            r_list = np.sort(np.array([closest_inds["RFrontal"],
                                      closest_inds["RParietal"]]))
            lr_verts = np.concatenate(stcs[0].vertices)
            l_list = [lr_verts[v] for v in l_list]
            r_list = [lr_verts[v] for v in r_list]
            vert_list[0], vert_list[1] = l_list, r_list
            stc_data = np.ones(len(np.concatenate(vert_list)))
            mask_stc = mne.MixedSourceEstimate(stc_data, vert_list, 0, 1,
                                               subject=subj_str)
            mask_fwd = mne.forward.restrict_forward_to_stc(mix_fwd, mask_stc)

            sp_stcs = mixed_norm(evos, mask_fwd, cov, alpha="sure",
                                 pick_ori="vector")

            gain = mask_fwd["sol"]["data"]
            gain = np.stack([gain[:, 0::3], gain[:, 1::3], gain[:, 2::3]])
            gain = np.linalg.norm(gain, axis=0)
            src_n = gain.shape[1]
            sparse_combs = list(combinations(src_n, 2))
            sparse_mat = np.zeros((src_n, src_n))
            for sc in sparse_combs:
                sparse_mat[sc[0], sc[1]] = principle_angle(gain[:, sc[0]],
                                                           gain[:, sc[1]])
            plt.imshow(sparse_mat)
            breakpoint()
