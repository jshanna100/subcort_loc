from tf_mxne_mix import simulate_mix_stc, stc_to_epo, tf_mxne_sub
import numpy as np
import mne
from mne.datasets import sample
from os.path import join
import matplotlib.pyplot as plt
from mne.viz import plot_sparse_source_estimates
from mne.inverse_sparse.subspace_pursuit import subspace_pursuit, subspace_pursuit_level
plt.ion()

root_dir = "/home/jev/"
# Import sample data
data_path = sample.data_path()
subjects_dir = join(data_path, 'subjects')
subject = 'sample'
cov_fname = join(data_path, "MEG", subject, 'sample_audvis-cov.fif')
raw_fname = join(data_path, 'MEG', subject, 'sample_audvis_raw.fif')
trans = join(data_path, 'MEG', subject, 'sample_audvis_raw-trans.fif')
bem_dir = join(subjects_dir, "sample",  'bem')
fname_bem = join(bem_dir, 'sample-5120-5120-5120-bem-sol.fif')
fwd_fname = join(data_path, 'MEG', subject,
                 'sample_audvis-meg-eeg-oct-6-fwd.fif')

# load cortical forward, convert to fixed orientations
ctx_fwd = mne.read_forward_solution(fwd_fname)
ctx_fwd = mne.convert_forward_solution(ctx_fwd, force_fixed=True)

# load subcortical patch forward
sub_fwd_file = "sample_sub-fwd.fif"
sub_fwd = mne.read_forward_solution(join(root_dir, "temp", sub_fwd_file))

ctx_fwd.pick_channels(sub_fwd.ch_names)

# we need mixed source space for simulation
fwd = mne.read_forward_solution(join(root_dir, "temp", "sample_mix-fwd.fif"))
mix_src = fwd["src"]
cov = mne.read_cov(cov_fname)
info = mne.io.read_info(raw_fname)

# get subcortical label(s)
subctx_labels = mne.get_volume_labels_from_src(mix_src, "sample", subjects_dir)
subctx_label = [lab for lab in subctx_labels if lab.name=="Hippocampus-lh"][0]
# get cortical labels
ctx_labels = []
ifg_labels = mne.read_labels_from_annot(subject, regexp='pars.*-lh',
                                        subjects_dir=subjects_dir)
if len(ifg_labels) != 3:
    raise ValueError("Should be 3 IFG labels.")
ifg_label = ifg_labels[0] + ifg_labels[1] + ifg_labels[2]
ctx_labels.append(ifg_label)
ctx_labels.append(mne.read_labels_from_annot(subject,
                                             regexp='supramarginal-lh',
                                             subjects_dir=subjects_dir)[0])
labels = ctx_labels + [subctx_label]
#labels = ctx_labels

# define parameters of signals
sig_params = {"freqs":[6., 6., 6.], "phases":[0, np.pi/2, np.pi/4], "length":900.,
              "event_length":.5, "n_events":150, "amp":10e-9, "mindist":3.,
              "sfreq":info["sfreq"], "jitters":[np.pi/12, np.pi/12, np.pi/12]}
# sig_params = {"freqs":[6., 6.], "phases":[0, np.pi/2], "length":900.,
#               "event_length":.5, "n_events":150, "amp":10e-9, "mindist":3.,
#               "sfreq":info["sfreq"], "jitters":[np.pi/12, np.pi/12]}

truth_stc, events, e_signals = simulate_mix_stc(sig_params, labels, [0, 0, 4], fwd)
epo = stc_to_epo(truth_stc, fwd, info, cov, events)

cnx = {"method":"wpli", "fmin":5, "fmax":7, "sfreq":epo.info["sfreq"]}
method = "ss"
snr = 1.
lambda2 = 1. / snr ** 2
ctx_verts = np.hstack(truth_stc.vertices[:2])
rrs = np.r_[fwd["src"][0]["rr"], fwd["src"][1]["rr"]][ctx_verts] * 170
if method == "tf_mxne":
    mix_stc, ctx_stc = tf_mxne_sub(ctx_fwd, sub_fwd, epo.average(), cov, trans, fname_bem,
                     25, .05, .3, 2, 10, .05, .3, 3, lambda2)
    fig = plot_sparse_source_estimates(ctx_fwd["src"], ctx_stc)
    fig._plotter.add_points(rrs, point_size=25, render_points_as_spheres=True)
elif method == "ss":
    fwd0_file = "sample_p_ico1-fwd.fif"
    fwd0 = mne.read_forward_solution(join(root_dir, "temp", fwd0_file))
    #fwd0 = None
    ctx_stc, fwds, resid = subspace_pursuit("sample", ["ico1", "ico2", "ico3"], fname_bem,
                                        epo, cov, trans, [1, 1, 1], lambda2,
                                        fwd0=fwd0, return_as_dipoles=False,
                                        subjects_dir=subjects_dir, mu=.5,
                                        sub_fwd=sub_fwd, sub_mu=.8, cnx=cnx,
                                        return_fwds=True, n_jobs=16,
                                        patch_comp_n=.9)
    #subcortical
    mix_src = fwds[-1]["src"] + sub_fwd["src"]
    mix_fwd = mne.make_forward_solution(epo.info, trans, mix_src, fname_bem)
    mix_fwd = mne.convert_forward_solution(mix_fwd, force_fixed=True)
    mix_gain = np.hstack((fwds[-1]["sol"]["data"], sub_fwd["sol"]["data"]))
    mix_fwd["sol"]["data"] = mix_gain
    mix_stc, est_fwd, var_expl, resid = subspace_pursuit_level(mix_fwd, epo, cov,
                                                               3, .5, lambda2)


    plt.figure()
    plt.plot(e_signals.T)
    print(mix_stc[0].vertices)

    fig = plot_sparse_source_estimates(fwd["src"], list(ctx_stc))
    fig._plotter.add_points(rrs, point_size=25, render_points_as_spheres=True)

    plt.figure()
    data = np.mean([cs.data for cs in ctx_stc], axis=0)
    plt.plot(data.T)

    plt.figure()
    data = np.mean([cs.data for cs in mix_stc], axis=0)
    plt.plot(data.T)

# ss_brain = stc.plot(subjects_dir=subjects_dir)
# for vtx in ctx_verts:
#     ss_brain.add_foci(vtx, coords_as_verts=True, color="red", alpha=0.3)
# for vtx in top_verts:
#     ss_brain.add_foci(vtx, coords_as_verts=True, color="blue", alpha=0.3)
