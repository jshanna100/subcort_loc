import mne
from os.path import join
from os import listdir
from mne.inverse_sparse.subspace_pursuit import subspace_pursuit
from mne.inverse_sparse import mixed_norm, make_stc_from_dipoles, gamma_map
from mne.minimum_norm import make_inverse_operator, apply_inverse
import numpy as np
from mne.viz import (plot_dipole_amplitudes, plot_dipole_locations,
                     plot_sparse_source_estimates)

root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
data_dir = join(root_dir, mem_dir, "02_MemTask")

subjs = ["120"]
sessions = ["2"]
preposts = ["pre", "post"]

subjects_dir = root_dir + "/freesurfer/subjects"
spacing = "ico5"
sc_base = ["Caudate", "Putamen", "Hippocampus", "Amygdala"]
sc_names = [f"Left-{x}" for x in sc_base] +  [f"Right-{x}" for x in sc_base]

snr = 3.6
lambda2 = 1. / snr**2

for subj in subjs:
    subj_str = f"MT-YG-{subj}"
    subj_dir = join(data_dir, subj_str)
    # cortex
    bem_file = f"MT-YG-{subj}-ctx-bem.fif"
    if bem_file in listdir(subj_dir):
        bem = mne.read_bem_solution(join(subj_dir, bem_file))
    else:
        bem_model = mne.make_bem_model(subj_str, subjects_dir=subjects_dir,
                                       ico=4)
        bem = mne.make_bem_solution(bem_model)
        mne.write_bem_solution(join(subj_dir, bem_file), bem)

    for sess in sessions:
        sess_dir = join(subj_dir, f"Session{sess}", "EEG")
        trans = join(sess_dir, f"MT-YG-{subj}_Session{sess}-trans.fif")
        epo = mne.read_epochs(join(sess_dir, f"MT-YG-{subj}_pre{sess}-epo.fif"),
                              preload=True)
        epo = epo["F_peak"]
        epo.crop(tmin=-.25, tmax=.25)
        cov = mne.compute_covariance(epo, keep_sample_mean=False)
        evo = epo.average()
        fwd0_file = f"MT-YG-{subj}_Session{sess}_p_ico1-fwd.fif"
        fwd0 = mne.read_forward_solution(join(sess_dir, fwd0_file))
        #fwd0 = None
        ss_out, fwd0 = subspace_pursuit(subj_str, ["ico1", "ico2", "ico3"], bem,
                                        evo, cov, trans, 8, lambda2, fwd0=fwd0,
                                        return_fwd0=True, n_jobs=16)

        fwd_file = f"MT-YG-{subj}_Session{sess}-ctx-fwd.fif"
        fwd = mne.read_forward_solution(join(sess_dir, fwd_file))
        # mixed norm
        loose, depth = 0.9, 0.9
        inverse_operator = make_inverse_operator(evo.info, fwd, cov,
                                                 depth=depth, fixed=True,
                                                 use_cps=True)
        stc_dspm = apply_inverse(evo, inverse_operator, lambda2=lambda2,
                                 method='dSPM')
        mxne_out = mixed_norm(evo, fwd, cov, weights=stc_dspm,
                              alpha=10, return_as_dipoles=False)

        gamma_out = gamma_map(evo, fwd, cov, 0.1)
