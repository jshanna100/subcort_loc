import mne
from mne.inverse_sparse import mixed_norm, gamma_map
from mne.minimum_norm import make_inverse_resolution_matrix as make_res
from mne.time_frequency import tfr_array_morlet
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from mne.forward.forward import _restrict_forward_to_src_sel
from itertools import combinations, product
from scipy.stats import circmean
from circular_hist import circ_hist_norm
plt.ion()

def norm_rm(rm):
    stack_rm = np.stack((rm[0::3, 0::3], rm[1::3, 1::3], rm[2::3, 2::3]))
    norm = np.linalg.norm(stack_rm, axis=0)
    return norm

root_dir = "/home/jev/"
subjects_dir = join(root_dir, "freesurfer", "subjects")
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
data_dir = join(root_dir, mem_dir, "02_MemTask")

snr = 3.0
lambda2 = 1.0 / snr ** 2

sp_method = mixed_norm

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
            epo.crop(tmin=-.25, tmax=.25)
            ctx_fwd = mne.read_forward_solution(join(sess_dir,
                                                f"MT-YG-{subj}_Session{sess}-ctx-fwd.fif"))
            mix_fwd = mne.read_forward_solution(join(sess_dir,
                                                f"MT-YG-{subj}_Session{sess}-mix-fwd.fif"))

            # restrict to sensitive parts of cortex
            sens = mne.sensitivity_map(ctx_fwd, ch_type="eeg")
            v_n = len(sens.vertices[0])
            ctx_lh_verts = sens.vertices[0][(sens.data[:v_n]>.3)[:,0]]
            ctx_rh_verts = sens.vertices[1][(sens.data[v_n:]>.3)[:,0]]
            sub_verts = [s["vertno"] for s in mix_fwd["src"][2:]]
            stc_data = np.ones(len(ctx_lh_verts) + len(ctx_rh_verts))
            temp_stc = mne.SourceEstimate(stc_data,
                                          [ctx_lh_verts, ctx_rh_verts],
                                          0, 1, subject=subj_str)
            ctx_fwd = mne.forward.restrict_forward_to_stc(ctx_fwd, temp_stc)
            verts = [ctx_lh_verts, ctx_rh_verts]
            verts.extend(sub_verts)
            stc_data = np.ones(sum([len(v) for v in verts]))
            temp_stc = mne.MixedSourceEstimate(stc_data, verts, 0, 1,
                                               subject=subj_str)
            mix_fwd = mne.forward.restrict_forward_to_stc(mix_fwd, temp_stc)

            for frontpar in ["F_peak", "P_peak"]:
                e = epo[frontpar]
                cov = mne.compute_covariance(e, keep_sample_mean=False)
                evo = e.average()
                # first resolution matrix
                inv_op = mne.minimum_norm.make_inverse_operator(info=epo.info,
                                                                forward=mix_fwd,
                                                                noise_cov=cov,
                                                                depth=None)
                rm_big = make_res(mix_fwd, inv_op, method='MNE', lambda2=lambda2)
                rm_big = norm_rm(rm_big)

                stc = sp_method(evo, ctx_fwd, cov, alpha=30)

                # display constellation of sources
                brain = stc.plot(subjects_dir=subjects_dir, hemi="both",
                                 title=f"{subj} {sess} {pp} {frontpar}")
                verts = np.concatenate((stc.lh_vertno, stc.rh_vertno))
                for v_idx, v in enumerate(verts):
                    hemi = "lh" if v_idx < len(stc.lh_vertno) else "rh"
                    brain.add_foci(v, coords_as_verts=True, color="black",
                                   hemi=hemi, scale_factor=0.5)
                vert_list = [s["vertno"] for s in mix_fwd["src"]]
                vert_list[0] = stc.lh_vertno
                vert_list[1] = stc.rh_vertno

                stc_data = np.ones(len(np.concatenate(vert_list)))
                mask_stc = mne.MixedSourceEstimate(stc_data, vert_list, 0, 1,
                                                   subject=subj_str)
                mask_fwd = mne.forward.restrict_forward_to_stc(mix_fwd, mask_stc)

                # sparse cortical resolution matrix
                inv_op = mne.minimum_norm.make_inverse_operator(info=epo.info,
                                                                forward=mask_fwd,
                                                                noise_cov=cov,
                                                                depth=None)
                rm_sp_ctx = make_res(mask_fwd, inv_op, method='MNE', lambda2=lambda2)
                rm_sp_ctx = norm_rm(rm_sp_ctx)
                plt.figure()
                plt.imshow(rm_sp_ctx, vmin=-0.05, vmax=0.05)
                plt.title(f"Sparse cortex {subj} {sess} {pp} {frontpar}")

                sp_stc = sp_method(evo, mask_fwd, cov, alpha=30,
                                    pick_ori="vector")


                # sparse - sparse cortical resolution matrix
                ## first mask the forward model again
                verts = sp_stc.vertices
                stc_data = np.ones(len(np.concatenate(verts)))
                mask_stc = mne.MixedSourceEstimate(stc_data, verts, 0, 1,
                                                   subject=subj_str)
                sp_fwd = mne.forward.restrict_forward_to_stc(mask_fwd, mask_stc)
                inv_op = mne.minimum_norm.make_inverse_operator(info=epo.info,
                                                                forward=sp_fwd,
                                                                noise_cov=cov,
                                                                depth=None)
                rm_sp = make_res(sp_fwd, inv_op, method='MNE', lambda2=lambda2)
                rm_sp = norm_rm(rm_sp)
                plt.figure()
                plt.imshow(rm_sp)
                plt.title(f"Sparse {subj} {sess} {pp} {frontpar}")
                breakpoint()
