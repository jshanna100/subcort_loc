import numpy as np
from functools import partial
from os.path import join
import mne
from mne.source_space import _set_source_space_vertices
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              apply_inverse_epochs)
from mne.inverse_sparse import tf_mixed_norm
from mne.simulation import simulate_sparse_stc
from principle_angles import *
from mne.io.pick import _picks_by_type

def generate_osc(freqs=None, phases=None, length=None, event_length=None,
                 n_events=None, amp=None, mindist=None, sfreq=None,
                 jitters=None):
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

def simulate_mix_stc(sig_params, labels, label_source_inds, fwd):
    signals, sig_events, e_signals = generate_osc(**sig_params)
    # split to random vector of 3
    data = []
    for signal in signals:
        data.append(split_to_vector(signal))
    data = np.swapaxes(np.array(data), 1, 2)
    # Define when the activity occurs using events.
    events = np.zeros((len(sig_events), 3), int)
    events[:len(events), 0] = sig_events
    events[:len(events), 2] = 1
    if len(data) != len(labels):
        raise ValueError("Number of signals must match numbeer of labels.")

    # random vertex from each label
    vertices = [[] for s in range(len(fwd["src"]))]
    for label, src_idx in zip(labels, label_source_inds):
        lab_intersect = np.intersect1d(fwd["src"][src_idx]["vertno"],
                                       label.vertices)
        vtx = np.random.choice(lab_intersect)
        vertices[src_idx].append(vtx)
    vertices = [np.sort(vert) for vert in vertices]

    tstep = 1. / sig_params["sfreq"]
    mix_stc = mne.MixedVectorSourceEstimate(data, vertices=vertices,
                                            tmin=0., tstep=tstep,
                                            subject=fwd["src"]._subject)

    return mix_stc, events, e_signals

def stc_to_epo(stc, fwd, info, cov, events):
    raw = mne.apply_forward_raw(fwd, stc, info)
    raw.pick_types(eeg=True)
    raw.set_eeg_reference(projection=True)
    cov.pick_channels(raw.ch_names)

    # Simulate
    raw_noi = raw.copy()
    mne.simulation.add_noise(raw_noi, cov)
    raw_noi.filter(l_freq=1, h_freq=20)
    epo = mne.Epochs(raw_noi, events, tmin=0, tmax=0.5, baseline=None)

    return epo

def whiten_gain(gain, info):
    pre_whitener = np.empty([len(gain), 1])
    for _, picks_ in _picks_by_type(info, ref_meg=False, exclude=[]):
        pre_whitener[picks_] = np.std(gain[picks_])
    wh_gain = gain / pre_whitener
    return wh_gain, pre_whitener

def calc_confl_mat(wh_gain, mu):
    # conflict matrix for mutual coherence of gain matrix columns
    src_n = wh_gain.shape[1]
    confl_mat = np.zeros((src_n, src_n), dtype=bool)
    norms = np.linalg.norm(wh_gain, axis=0)
    inner = np.inner(wh_gain.T, wh_gain.T)
    for src_i in range(src_n):
        for src_j in range(src_i+1, src_n):
            if inner[src_i, src_j] > (mu * norms[src_i] * norms[src_j]):
                confl_mat[src_i, src_j] = 1
    return confl_mat

def select_s(proxy, confl_mat, s):
    """
    Get s strongest sources, with the constraint that mutual coherence
    is below threshold
    """
    if len(proxy.shape) == 3:
        proxy = proxy.mean(axis=0)

    norm_sort_inds = np.argsort(np.linalg.norm(proxy, axis=1))[::-1]
    support = np.array([norm_sort_inds[0]])
    for idx in norm_sort_inds[1:]:
        if len(support) == s:
            break
        if sum(confl_mat[support, idx]) == 0:
            support = np.append(support, idx)
        else:
            print("Conflict avoided.")

    support = np.unique(support)
    return support

def restrict_stc_vertices(stc, gain_inds):
    # restrict stc to those vertices
    data_vert_src = []
    for s_idx, verts in enumerate(stc.vertices):
        for v in verts:
            data_vert_src.append((v, s_idx))
    new_vertices = [[] for x in range(len(stc.vertices))]
    for tv in gain_inds:
        (vtx, src_idx) = data_vert_src[tv]
        new_vertices[src_idx].append(vtx)
    new_vertices = [np.sort(np.array(nv).astype(int)) for nv in new_vertices]
    stc.vertices = new_vertices
    stc.data = stc.data[gain_inds]
    return stc

def tf_mxne_sub(ctx_fwd, sub_fwd, inst, cov, trans, fname_bem,
                ctx_alpha, ctx_l1_ratio, ctx_mu, ctx_top,
                mix_alpha, mix_l1_ratio, mix_mu, mix_top,
                lambda2):
    if isinstance(inst, mne.BaseEpochs):
        evo = inst.average()
    elif isinstance(inst, mne.Evoked):
        evo = inst
    else:
        raise ValueError("Instance must be Epochs or Evoked.")
    #tf mixed-norm
    ctx_mxtf = tf_mixed_norm(evo, ctx_fwd, cov, l1_ratio=ctx_l1_ratio,
                             alpha=ctx_alpha, loose=0, depth=0)
    ## top cortical sources
    # restrict gain matrix to those vertices found by the tf_mixed_norm
    temp_fwd = mne.forward.restrict_forward_to_stc(ctx_fwd, ctx_mxtf)
    gain = temp_fwd["sol"]["data"]
    # calculate the conflix matrix to exclude coherent sources
    wh_gain, _ = whiten_gain(gain, ctx_fwd["info"])
    confl_mat = calc_confl_mat(wh_gain, ctx_mu)
    top_inds = select_s(ctx_mxtf.data, confl_mat, ctx_top)

    # restrict stc to those vertices
    ctx_mxtf = restrict_stc_vertices(ctx_mxtf, top_inds)
    sp_fwd = mne.forward.restrict_forward_to_stc(ctx_fwd, ctx_mxtf)

    # make mixed forward
    mix_src = sp_fwd["src"] + sub_fwd["src"]
    mix_fwd = mne.make_forward_solution(inst.info, trans, mix_src, fname_bem)
    mix_fwd = mne.convert_forward_solution(mix_fwd, force_fixed=True)
    # need to fix gain matrix because sub_fwd is an SVD patch forward model
    mix_gain = np.hstack((sp_fwd["sol"]["data"], sub_fwd["sol"]["data"]))
    mix_fwd["sol"]["data"] = mix_gain

    # localise on mixed space
    mix_mxtf = tf_mixed_norm(evo, mix_fwd, cov, l1_ratio=mix_l1_ratio,
                             alpha=mix_alpha, loose=0, depth=0)
    # top mixed sources

    # restrict gain matrix to those vertices found by the tf_mixed_norm
    temp_fwd = mne.forward.restrict_forward_to_stc(mix_fwd, mix_mxtf)
    gain = temp_fwd["sol"]["data"]
    # calculate the conflix matrix to exclude coherent sources
    wh_gain, _ = whiten_gain(gain, ctx_fwd["info"])
    confl_mat = calc_confl_mat(wh_gain, ctx_mu)
    top_inds = select_s(mix_mxtf.data, confl_mat, mix_top)

    # restrict stc to those vertices
    mix_mxtf = restrict_stc_vertices(mix_mxtf, top_inds)
    sp_mix_fwd = mne.forward.restrict_forward_to_stc(mix_fwd, mix_mxtf)

    inverse_operator = make_inverse_operator(inst.info, sp_mix_fwd, cov,
                                             use_cps=True, loose=0, depth=0)

    if isinstance(inst, mne.BaseEpochs):
        stc_mne = apply_inverse_epochs(inst, inverse_operator,
                                       lambda2=lambda2,
                                       method='MNE')
    elif isinstance(inst, mne.Evoked):
        stc_mne = apply_inverse(inst, inverse_operator, lambda2=lambda2,
                                method='MNE')

    return stc_mne, ctx_mxtf
