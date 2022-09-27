import numpy as np
from os.path import join
import mne
from mne.datasets import sample
from mne.inverse_sparse import gamma_map, mixed_norm, make_stc_from_dipoles
from mne.inverse_sparse.subspace_pursuit import subspace_pursuit
from mne.viz import (plot_sparse_source_estimates,
                     plot_dipole_locations, plot_dipole_amplitudes)
from mne.minimum_norm import make_inverse_operator, apply_inverse

print(__doc__)

data_path = sample.data_path()
subjects_dir = join(data_path, 'subjects')
meg_path = join(data_path, 'MEG', 'sample')
fwd_fname = join(meg_path, 'sample_audvis-meg-eeg-oct-6-fwd.fif')
evoked_fname = join(meg_path, 'sample_audvis-ave.fif')
cov_fname = join(meg_path, 'sample_audvis-shrunk-cov.fif')
trans = join(meg_path, 'sample_audvis_raw-trans.fif')
bem_dir = join(subjects_dir, "sample",  'bem')
fname_bem = join(bem_dir, 'sample-5120-5120-5120-bem-sol.fif')

# Read the evoked response and crop it
condition = 'Left visual'
evoked = mne.read_evokeds(evoked_fname, condition=condition,
                          baseline=(None, 0))
evoked = evoked.pick_types(meg="mag")
evoked.crop(tmin=-50e-3, tmax=300e-3)
# get noise covariance matrix
cov = mne.read_cov(cov_fname)
cov = cov.pick_channels(evoked.ch_names)

# Read the forward solution
forward = mne.read_forward_solution(fwd_fname)

# Run the Gamma-MAP method with dipole output
gamma_out = gamma_map(evoked, forward, cov, 0.15, xyz_same_gamma=True,
                      return_as_dipoles=True)
gamma_stc = make_stc_from_dipoles(gamma_out, forward["src"])

# mixed norm
loose, depth = 0.9, 0.9
inverse_operator = make_inverse_operator(evoked.info, forward, cov,
                                         depth=depth, fixed=True,
                                         use_cps=True)
stc_dspm = apply_inverse(evoked, inverse_operator, lambda2=1. / 9.,
                         method='dSPM')
mxne_out = mixed_norm(evoked, forward, cov, weights=stc_dspm,
                      alpha=10, return_as_dipoles=True)
mxne_stc = make_stc_from_dipoles(mxne_out, forward["src"])


fwd0_file = "sample_p_ico1-fwd.fif"
fwd0 = mne.read_forward_solution(join("/home", "jev", "temp", fwd0_file))

# inverse_operator = make_inverse_operator(evoked.info, fwd0, cov,
#                                          depth=None, fixed=True, loose=0)
# stc_mne = apply_inverse(evoked, inverse_operator, lambda2=1. / 15.,
#                          method='MNE')
# breakpoint()
ss_out, fwd0 = subspace_pursuit("sample", ["ico1", "ico2", "ico3"], fname_bem,
                                evoked, cov, trans, 4, 1/9, fwd0=None,
                                return_as_dipoles=False,
                                return_fwd0=True, n_jobs=16)

# mne.write_forward_solution(join("/home", "jev", "temp", fwd0_file),
#                            fwd0, overwrite=True)
