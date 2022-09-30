# Author: Kostiantyn Maksymenko <kostiantyn.maksymenko@gmail.com>
#
# License: BSD (3-clause)
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from os.path import join
import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.inverse_sparse import gamma_map, mixed_norm, make_stc_from_dipoles
from mne.inverse_sparse.subspace_pursuit import subspace_pursuit

random_state = 42  # set random state to make this example deterministic

# Import sample data
data_path = sample.data_path()
subjects_dir = join(data_path, 'subjects')
subject = 'sample'
trans = join(data_path, 'MEG', subject, 'sample_audvis_raw-trans.fif')
raw_fname = join(data_path, 'MEG', subject, 'sample_audvis_raw.fif')
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

subctx_labels = mne.get_volume_labels_from_src(vol_src, "sample", subjects_dir)
left_hipp_label = [lab for lab in subctx_labels if lab.name=="Hippocampus-lh"][0]

# Import forward operator and source space
fwd_fname = join(data_path, 'MEG', subject,
                 'sample_audvis-meg-eeg-oct-6-fwd.fif')
fwd = mne.read_forward_solution(fwd_fname)
src = fwd['src']

mix_src = src + vol_src
mix_fwd = mne.make_forward_solution(info, trans, mix_src, fname_bem,
                                    n_jobs=16)

# To select source, we use the caudal middle frontal to grow
# a region of interest.
selected_label = mne.read_labels_from_annot(
    subject, regexp='caudalmiddlefrontal-lh', subjects_dir=subjects_dir)[0]

# WHERE?

location = "center"  # Use the index of the vertex as a seed
extent = 0.  # One dipole source
label_dipole = mne.label.select_sources(
    subject, left_hipp_label, location=location, extent=extent,
    subjects_dir=subjects_dir, random_state=random_state)

# WHAT?
# Define the time course of the activity
source_time_series = np.sin(2. * np.pi * 18. * np.arange(100) * tstep) * 10e-9

# WHEN?
# Define when the activity occurs using events.
n_events = 50
events = np.zeros((n_events, 3), int)
events[:, 0] = 200 * np.arange(n_events)  # Events sample.
events[:, 2] = 1  # All events have the sample id.

# Set up simulators
source_simulator = mne.simulation.SourceSimulator(mix_src, tstep=tstep)
source_simulator.add_data(label_dipole, source_time_series, events)

# Simulate
raw = mne.simulation.simulate_raw(info, source_simulator, forward=mix_fwd)
raw = raw.pick_types(meg=False, eeg=True, stim=True)
cov = mne.make_ad_hoc_cov(raw.info)
mne.simulation.add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.04],
                         random_state=random_state)

# evoked
events = mne.find_events(raw, initial_event=True)
tmax = (len(source_time_series) - 1) * tstep
epochs = mne.Epochs(raw, events, 1, tmin=0, tmax=tmax, baseline=None)
evoked = epochs.average()

# ground truth in stc form
stc_true = source_simulator.get_stc(start_sample=0,
                                    stop_sample=len(source_time_series))

# # subspace pursuit
# fwd0_file = "sample_p_ico1-fwd.fif"
# fwd0 = mne.read_forward_solution(join("/home", "jev", "temp", fwd0_file))
# ss_out, fwd0 = subspace_pursuit("sample", ["ico1", "ico2", "ico3"], fname_bem,
#                                 evoked, cov, trans, 1, 1/9, fwd0=fwd0,
#                                 return_as_dipoles=False,
#                                 return_fwd0=True, n_jobs=16)
# ss_out.plot()

# mixed norm
loose, depth = 0.9, 0.9
inverse_operator = make_inverse_operator(evoked.info, fwd, cov,
                                         depth=depth, fixed=True,
                                         use_cps=True)
stc_dspm = apply_inverse(evoked, inverse_operator, lambda2=1. / 9.,
                         method='dSPM')
mxne_out = mixed_norm(evoked, fwd, cov, weights=stc_dspm,
                      alpha=50, return_as_dipoles=True)
mxne_stc = make_stc_from_dipoles(mxne_out, fwd["src"])
mxne_stc.plot()

# gamma map
gamma_out = gamma_map(evoked, fwd, cov, 0.7, xyz_same_gamma=True,
                      return_as_dipoles=True)
gamma_stc = make_stc_from_dipoles(gamma_out, fwd["src"])
gamma_stc.plot()

stc_true.plot()
