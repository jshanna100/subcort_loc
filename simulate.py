# Author: Kostiantyn Maksymenko <kostiantyn.maksymenko@gmail.com>
#
# License: BSD (3-clause)
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from functools import partial
from os.path import join
import mne
from mne.source_space import _set_source_space_vertices
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.inverse_sparse import (gamma_map, mixed_norm, make_stc_from_dipoles,
                                tf_mixed_norm)
from mne.inverse_sparse.subspace_pursuit import (subspace_pursuit,
                                                 make_patch_forward,
                                                 subspace_pursuit_level)
from mne.simulation import simulate_sparse_stc
from principle_angles import *

def generate_osc(freqs, phases, length, event_length, n_events, amp, mindist,
                 sfreq, jitters=None):
    signals = [np.zeros(int(length * sfreq)) for x in range(len(freqs))]
    times = np.linspace(0, event_length, int(event_length * sfreq))
    e_signals = []
    if jitters is None:
        jitters = [0 for x in range(len(signals))]
    # e signals are the canonical signals without jitter
    for signal, freq, phase in zip(signals, freqs, phases):
        e_signal = np.sin(2. * np.pi * freq * times + phase) * amp
        hanning = np.hanning(len(e_signal))
        e_signal *= hanning
        e_signals.append(e_signal)
    e_signals = np.array(e_signals)
    # generate events
    events = np.sort(np.random.randint(len(signals[0]), size=n_events))
    # no events too close together
    dists = events[1:] - events[:-1]
    bad_events = np.where((dists*sfreq) < mindist)[0]
    events = np.delete(events, bad_events+1)
    for eve in events:
        if (len(signals[0]) - eve) > len(times):
            for signal, freq, phase, jitter in zip(signals, freqs, phases, jitters):
                jitter = 2 * jitter * np.random.random_sample() - jitter
                this_signal = np.sin(2. * np.pi * freq * times + phase + jitter) * amp
                hanning = np.hanning(len(this_signal))
                this_signal *= hanning
                signal[eve:eve+len(e_signal)] = this_signal
    signals = np.array(signals)
    return signals, events, e_signals

def split_to_vector(series):
    weights = np.random.uniform(size=3)
    weights = weights / np.linalg.norm(weights)
    new_series = np.dot(series[:, None], weights[None, :])
    return new_series

root_dir = "/home/hannaj/"
root_dir = "/home/jev/"
random_state = 42  # set random state to make this example deterministic
angle_coef = .8

# Import sample data
data_path = sample.data_path()
subjects_dir = join(data_path, 'subjects')
subject = 'sample'
trans = join(data_path, 'MEG', subject, 'sample_audvis_raw-trans.fif')
raw_fname = join(data_path, 'MEG', subject, 'sample_audvis_raw.fif')
cov_fname = join(data_path, "MEG", subject, 'sample_audvis-cov.fif')
info = mne.io.read_info(raw_fname)
tstep = 1. / info['sfreq']
bem_dir = join(subjects_dir, "sample",  'bem')
fname_bem = join(bem_dir, 'sample-5120-5120-5120-bem-sol.fif')

# sub cortical
sc_base = ["Caudate", "Putamen", "Hippocampus", "Amygdala"]
sc_names = [f"Left-{x}" for x in sc_base] +  [f"Right-{x}" for x in sc_base]
vol_src = mne.setup_volume_source_space("sample", bem=fname_bem,
                                        volume_label=sc_names,
                                        subjects_dir=subjects_dir)



# # Import forward operator and source space
fwd_fname = join(data_path, 'MEG', subject,
                 'sample_audvis-meg-eeg-oct-6-fwd.fif')
fwd = mne.read_forward_solution(fwd_fname)
src = fwd['src']
fwd_file = "sample_mix-fwd.fif"
mix_src = src + vol_src
mix_fwd = mne.make_forward_solution(info, trans, mix_src, fname_bem,
                                    n_jobs=16)


# subcortical labels
subctx_labels = mne.get_volume_labels_from_src(mix_src, "sample", subjects_dir)
subctx_label = [lab for lab in subctx_labels if lab.name=="Hippocampus-lh"][0]

# cortical label
ctx_labels = []
ctx_labels.append(mne.read_labels_from_annot(subject,
                                             regexp='rostralmiddlefrontal-lh',
                                             subjects_dir=subjects_dir)[0])
ctx_labels.append(mne.read_labels_from_annot(subject,
                                             regexp='inferiorparietal-lh',
                                             subjects_dir=subjects_dir)[0])

# Define the time course of the activity
sig_len = 900.
event_n = 150
amp = 10e-9
signals, sig_events, e_signals = generate_osc([6., 6., 6.], [0, 0, np.pi/2],
                                              sig_len, 0.5, event_n, amp, 3.,
                                              info["sfreq"],
                                              jitters=[np.pi/16,
                                                       np.pi/16,
                                                       np.pi/16])
# signals, sig_events, e_signals = generate_osc([6., 6.,], [0, 0],
#                                               sig_len, 0.5, event_n, amp, 3.,
#                                               info["sfreq"],
#                                               jitters=[np.pi/16,
#                                                        np.pi/16])

# split to random vector of 3
data = []
for signal in signals:
    data.append(split_to_vector(signal))
data = np.swapaxes(np.array(data), 1, 2)
# Define when the activity occurs using events.
events = np.zeros((len(sig_events), 3), int)
events[:len(events), 0] = sig_events
events[:len(events), 2] = 1


# random vertex from each label
ctx_verts = []
for ctx_label in ctx_labels:
    ctx_use_verts = np.intersect1d(mix_src[0]["vertno"], ctx_label.vertices)
    ctx_verts.append(np.random.choice(ctx_use_verts))
ctx_verts = np.array(ctx_verts)
ctx_verts.sort()
subctx_use_verts = np.intersect1d(mix_src[4]["vertno"], subctx_label.vertices)
subctx_vtx = np.random.choice(subctx_use_verts, size=1)

vertices = [np.array([], dtype=int) for s in mix_src]
vertices[0] = ctx_verts
vertices[4] = subctx_vtx
mix_stc = mne.MixedVectorSourceEstimate(data, vertices=vertices, tmin=0.,
                                        tstep=tstep, subject="sample")
raw = mne.apply_forward_raw(mix_fwd, mix_stc, info)
raw.info["bads"].append("MEG 2443")
raw.pick_types(eeg=True)

# Simulate
cov = mne.read_cov(cov_fname)
raw_noi = raw.copy()
mne.simulation.add_noise(raw_noi, cov)
raw_noi.filter(l_freq=1, h_freq=20)

epo = mne.Epochs(raw, events, tmin=0, tmax=0.5, baseline=None)
evo1 = epo["1"].average()

epo_noi = mne.Epochs(raw_noi, events, tmin=0, tmax=0.5, baseline=None)
evo1_noi = epo_noi["1"].average()


snr = 0.8
lambda2 = 1. / snr ** 2
inst = evo1_noi
inst.set_eeg_reference(projection=True)
cov = cov.pick_channels(inst.ch_names)

# cnx = {"method":"wpli", "fmin":5, "fmax":7, "sfreq":inst.info["sfreq"]}
# #cnx = None
#
# # subspace pursuit
# fwd0_file = "sample_p_ico1-fwd.fif"
# fwd0 = mne.read_forward_solution(join(root_dir, "temp", fwd0_file))
# #fwd0 = None
#
# sub_fwd = make_patch_forward("sample", None, fname_bem, inst.info, trans,
#                              volume=True, volume_label=sc_names,
#                              subjects_dir=subjects_dir, n_jobs=16)
# sub_fwd_file = "sample_sub-fwd.fif"
# sub_fwd = mne.read_forward_solution(join(root_dir, "temp", sub_fwd_file))
# ss_out, fwds, resid = subspace_pursuit("sample", ["ico1", "ico2"], fname_bem,
#                                        inst, cov, trans, [1, 1], lambda2,
#                                        fwd0=fwd0, return_as_dipoles=False,
#                                        subjects_dir=subjects_dir, mu=.25,
#                                        sub_fwd=sub_fwd, sub_mu=.8, cnx=cnx,
#                                        return_fwds=True, n_jobs=16,
#                                        patch_comp_n=1)
#
# this_ss = ss_out[0].copy() if isinstance(ss_out, tuple) else ss_out.copy()
#
# out_d = [ss.data.copy() for ss in ss_out]
# out_d = np.array(out_d).mean(axis=0)
# out = this_ss
# out.data = out_d
#
# ss_brain = out.plot(subjects_dir=subjects_dir)
# for vtx in ctx_verts:
#     ss_brain.add_foci(vtx, coords_as_verts=True)

# mne.write_forward_solution(join(root_dir, "temp", fwd0_file),
#                            fwds[0], overwrite=True)

# subcortical
# mix_src = fwds[-1]["src"] + sub_fwd["src"]
# mix_fwd = mne.make_forward_solution(inst.info, trans, mix_src, fname_bem)
# mix_fwd = mne.convert_forward_solution(mix_fwd, force_fixed=True)
# mix_gain = np.hstack((fwds[-1]["sol"]["data"], sub_fwd["sol"]["data"]))
# mix_fwd["sol"]["data"] = mix_gain
# out, est_fwd, var_expl, resid = subspace_pursuit_level(mix_fwd, inst, cov,
#                                                        3, .5, lambda2)

# plt.ion()
# plt.figure()
# plt.plot(out.data.T)
# plt.figure()
# plt.plot(e_signals.T)



ss_brain = stc_mxtf.plot(subjects_dir=subjects_dir)
for vtx in ctx_verts:
    ss_brain.add_foci(vtx, coords_as_verts=True, color="red", alpha=0.3)
for vtx in top_verts:
    ss_brain.add_foci(vtx, coords_as_verts=True, color="blue", alpha=0.3)
