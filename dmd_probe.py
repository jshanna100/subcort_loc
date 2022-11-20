import mne
from os.path import join
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import zscore
import matplotlib.pyplot as plt
plt.ion()
from pydmd import OptDMD, MrDMD

root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
data_dir = join(root_dir, mem_dir, "02_MemTask")

subjs = ["120"]
sessions = ["2"]
preposts = ["pre", "post"]

for subj in subjs:
    subj_str = f"MT-YG-{subj}"
    subj_dir = join(data_dir, subj_str)
    for sess in sessions:
        for pp in preposts:
            sess_dir = join(subj_dir, f"Session{sess}", "EEG")
            raw = mne.io.Raw(join(sess_dir, f"MT-YG-{subj}_{pp}{sess}_ica-raw.fif"),
                             preload=True)
            raw.pick_types(eeg=True)
            all_chans = raw.ch_names
            #raw.set_eeg_reference(projection=True) # average reference
            raw.filter(l_freq=4, h_freq=7, picks=raw.ch_names)

            data = raw.get_data() * 1e+6
            mrdmd = MrDMD(OptDMD(), max_cycles=1, max_level=8)
            mrdmd.fit(data)

            recon_data = mrdmd.reconstructed_data.real * 1e-6
            recon_raw = mne.io.RawArray(recon_data, raw.info)
            breakpoint()
            mode_raws = []
            for level_idx in range(mrdmd.max_level):
                modes = mrdmd.partial_modes(level=level_idx).real
                if not len(dyn):
                    continue
                fig, axes = plt.subplots(1, len(modes), squeeze=False)
                for mode_idx in range(len(modes)):
                    ax = axes[0, mode_idx]
                    mne.viz.plot_topomap(modes[dyn_idx,], raw.info, axes=ax)
                plt.suptitle(f"Level {level_idx}")
                plt.tight_layout()
                dyns = mrdmd.partial_dynamics(level=level_idx).real.T * 1e+6
                info = mne.create_info(len(modes), sfreq=raw.info["sfreq"],
                                       ch_types="eeg")
                mode_raw = mne.io.RawArray(modes, info)
                mode_raws.append(mode_raw)
            breakpoint()
