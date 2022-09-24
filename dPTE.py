from mne.time_frequency.tfr import tfr_array_morlet
from mne.filter import filter_data
from scipy.signal import hilbert
from itertools import combinations
import numpy as np
from joblib import Parallel, delayed

def _get_delay(phase):
    """
    Computes the overall delay for all given channels

    Parameters
    ----------
    phase : numpy.ndarray
        m x n ndarray : m: number of channels, n: number of samples

    Returns
    -------
    delay : int
    """
    phase = phase
    m,n = phase.shape
    c1 = n*m
    r_phase = np.roll(phase, 1, axis=1)
    m = np.multiply(phase, r_phase)[1:-1]
    c2 = (m < 0).sum()
    delay = int(np.round(c1/c2))
    return delay

def _instant_phase(data, freqs, sfreq, n_cycles=None, method="wavelet",
                   freq_band=2, cuda=False, n_jobs=1):
    if method == "wavelet":
        phases = tfr_array_morlet(data, sfreq, freqs, n_cycles=n_cycles,
                                  output="phase", n_jobs=n_jobs)
    if method == "hilbert":
        phases = np.empty((data.shape[0], data.shape[1], len(freqs),
                           data.shape[2]))
        for freq_idx,freq in enumerate(list(freqs)):
            if cuda:
                temp_data = filter_data(data, sfreq, l_freq=freq-freq_band/2,
                                        h_freq=freq+freq_band/2, n_jobs="cuda")
            else:
                temp_data = filter_data(data, sfreq, l_freq=freq-freq_band/2,
                                        h_freq=freq+freq_band/2, n_jobs=n_jobs)
            analytic_signal = hilbert(temp_data)
            phases[:,:,freq_idx,:] = np.angle(analytic_signal)

    return phases


def _hillebrand_binnums(phase_series):
    # determine by formula given in Hillebrand et al 2016
    N = phase_series.size
    bin_nums = np.exp(0.626+0.4*np.log(N-1))
    return bin_nums

def _dPTE_pairwise(x_phase, y_phase, sfreq, delay, x_past_histo=None,
                   y_past_histo=None, epsilon=1e-8):
    ''' directed phase transfer entropy of x to y; to get y to x, just * -1 '''

    # we need the non-delayed and delayed phase for x and y
    phases = {}
    phases["x"] = x_phase[delay:]
    phases["y"] = y_phase[delay:]
    phases["x_past"] = x_phase[:-delay]
    phases["y_past"] = y_phase[:-delay]

    # calculate the bin edges
    bin_edges = {}
    for ph_k,ph_v in phases.items():
        #bw, bn = _adjust_binwidth(_scotts_binwidth(ph_v))
        bn = _hillebrand_binnums(ph_v)
        bin_edges[ph_k] = np.linspace(-np.pi, np.pi, num=int(np.round(bn)))

    # calculate the histograms
    samp_nums = phases["x"].size # x is arbitrary; any other variable would do
    if x_past_histo is None:
        x_past_histo = (np.histogram(phases["x_past"],
                        bins=bin_edges["x_past"])[0] / samp_nums + epsilon)
    if y_past_histo is None:
        y_past_histo = (np.histogram(phases["y_past"],
                        bins=bin_edges["y_past"])[0] / samp_nums + epsilon)
    x_past_x_histo = (np.histogramdd(np.array([phases["x_past"],
                      phases["x"]]).T, bins=[bin_edges["x_past"],
                      bin_edges["x"]])[0] / samp_nums + epsilon)
    y_past_y_histo = (np.histogramdd(np.array([phases["y_past"],
                      phases["y"]]).T, bins=[bin_edges["y_past"],
                      bin_edges["y"]])[0] / samp_nums + epsilon)
    x_past_y_past_histo = (np.histogramdd(np.array([phases["x_past"],
                           phases["y_past"]]).T,
                           bins=[bin_edges["x_past"],
                           bin_edges["y_past"]])[0] / samp_nums + epsilon)
    x_x_past_y_past_histo = (np.histogramdd(np.array([phases["x"],
                             phases["x_past"],
                             phases["y_past"]]).T,
                             bins=[bin_edges["x"], bin_edges["x_past"],
                             bin_edges["y_past"]])[0] / samp_nums + epsilon)
    y_y_past_x_past_histo = (np.histogramdd(np.array([phases["y"],
                             phases["y_past"],
                             phases["x_past"]]).T,
                             bins=[bin_edges["y"], bin_edges["y_past"],
                             bin_edges["x_past"]])[0] / samp_nums + epsilon)

    # entropy
    h_y_past_y = -1 * np.sum(y_past_y_histo * np.log(y_past_y_histo))
    h_y_past_x_past = -1 * np.sum(x_past_y_past_histo
                           * np.log(x_past_y_past_histo))
    h_y_past = -1 * np.sum(y_past_histo * np.log(y_past_histo))
    h_y_y_past_x_past = -1 * np.sum(y_y_past_x_past_histo
                         * np.log(y_y_past_x_past_histo))
    h_x_past_x = -1 * np.sum(x_past_x_histo * np.log(x_past_x_histo))
    h_x_past = -1 * np.sum(x_past_histo * np.log(x_past_histo))
    h_x_x_past_y_past = -1 * np.sum(x_x_past_y_past_histo
                         * np.log(x_x_past_y_past_histo))

    # PTE
    pte_xy = h_y_past_y + h_y_past_x_past - h_y_past - h_y_y_past_x_past
    pte_yx = h_x_past_x + h_y_past_x_past - h_x_past - h_x_x_past_y_past
    if pte_xy < 1e-5 and pte_yx < 1e-5:
        dPTE_xy = 0
    else:
        dPTE_xy = pte_xy / (pte_xy + pte_yx)

    return dPTE_xy

def _dPTE(phase, sfreq, delay=None, epsilon=1e-8, roi=None):
    ''' dPTE of all signals against all signals, phase should be signal*time '''
    dPTE_mat = np.zeros((phase.shape[0], phase.shape[0]))
    delay = _get_delay(phase)
    samp_nums = phase.shape[-1] - delay
    histos = []
    for idx in range(phase.shape[0]):
        bn = _hillebrand_binnums(phase[idx,:-delay])
        bin_edges = np.linspace(-np.pi, np.pi, num=int(bn))
        histos.append(np.histogram(phase[idx,:-delay],
                      bins=bin_edges)[0] / samp_nums + epsilon)
    if roi is None:
        combs = combinations(range(phase.shape[0]), 2)
    else:
        combs = [(roi, reg_idx) for reg_idx in range(phase.shape[0])]
    for (x, y) in combs:
        dPTE_mat[x, y] = _dPTE_pairwise(phase[x], phase[y], sfreq, delay,
                                        x_past_histo=histos[x],
                                        y_past_histo=histos[y])
    return dPTE_mat

def epo_dPTE(data, freqs, sfreq, delay=None, n_cycles=None, roi=None,
             phase_method="wavelet", freq_band=2, cuda=False, n_jobs=1):
    ''' dPTE on epoched data: epoch*signal*time
    data: e*c*t numpy array, e is epochs, c is channels/sources/t is samples
    freqs: array-like of frequencies
    sfreq: sampling rate
    delay: delay in ms, if None then automatically calculate
    phase_method: "wavelet" or "hilbert"
    freq_band: only relevant when phase_method is "hilbert"
    ROI: calculate dPTE only from this ROI idx

    ------

    returns e*c*c numpy array
    '''

    phase = _instant_phase(data, freqs, sfreq, n_cycles=n_cycles,
                           method=phase_method, freq_band=freq_band, cuda=cuda)
    phase = phase.mean(axis=2)
    results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(_dPTE)(phase[i,], sfreq, delay, roi=roi) for i in range(phase.shape[0]))
    dPTE = np.array(results)
    return dPTE
