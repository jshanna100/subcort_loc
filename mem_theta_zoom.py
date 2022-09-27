import mne
from os.path import join
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import zscore
import matplotlib.pyplot as plt
plt.ion()

def phase_align(peaks, phase, win_ind):
    new_peaks = []
    win_half = win_ind//2
    for peak in peaks:
        new_ind = np.argmax(phase[peak-win_half:peak+win_half])
        new_ind += peak-win_half
        new_peaks.append(new_ind)
    return np.array(new_peaks)

root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
data_dir = join(root_dir, mem_dir, "02_MemTask")

frontal_chans = ["F3", "F1", "Fz", "F2", "F4", "FC3", "FC1", "FC2", "FC4"]
parietal_chans = ["PO7", "PO3", "POz", "PO4", "PO8", "P5", "P1", "P2", "P6"]

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
            all_chans = raw.ch_names
            raw.set_eeg_reference(projection=True) # average reference
            #raw.set_channel_types({x:"eeg" for x in chck_chans})
            raw.filter(l_freq=4, h_freq=6.5, picks=raw.ch_names) # theta band
            # create frontal/parietal summary channels, centro-lateral for counterindication
            front_data = raw.get_data(picks=frontal_chans).mean(axis=0, keepdims=True)
            parietal_data = raw.get_data(picks=parietal_chans).mean(axis=0, keepdims=True)
            data = np.vstack([front_data, parietal_data])
            info = mne.create_info(["front", "parietal"],
                                   raw.info["sfreq"], ch_types="eeg")
            new_raw = mne.io.RawArray(data, info)
            epo = mne.make_fixed_length_epochs(new_raw, 10)
            tfr = mne.time_frequency.tfr_morlet(epo, [4, 4.5, 5, 5.5, 6, 6.5], 3,
                                                output="complex", return_itc=False,
                                                average=False)
            power = np.abs(tfr).mean(axis=2) * 1e+5
            power = zscore(power, axis=0)
            phase = np.angle(tfr).mean(axis=2)
            # tiresome work here to put this back in raw format
            power = np.transpose(power, [0, 2, 1])
            power = power.reshape(power.shape[0] * power.shape[1], power.shape[2])
            power = np.transpose(power, [1, 0])
            phase = np.transpose(phase, [0, 2, 1])
            phase = phase.reshape(phase.shape[0] * phase.shape[1], phase.shape[2])
            phase = np.transpose(phase, [1, 0])

            fp_sync = 1 - np.sin(np.abs(phase[0,] - phase[1,])/2)[None,]
            info = mne.create_info(["F_Power", "P_Power",
                                    "F_Phase", "P_Phase", "FP_Sync"],
                                    raw.info["sfreq"])
            pp_raw = mne.io.RawArray(np.concatenate([power, phase, fp_sync], axis=0),
                                     info)
            new_raw.crop(tmin=0, tmax=pp_raw.times[-1])
            new_raw.add_channels([pp_raw], force_update_info=True)

            pow_dat = new_raw.get_data(["F_Power", "P_Power"])
            prominence = 3
            win_ind = raw.time_as_index(.25)[0]
            f_peaks = find_peaks(pow_dat[0,], prominence=prominence)[0]
            f_peaks = phase_align(f_peaks, phase[0,], win_ind)
            p_peaks = find_peaks(pow_dat[1,], prominence=prominence)[0]
            p_peaks = phase_align(p_peaks, phase[1,], win_ind)

            for f in f_peaks:
                new_raw.annotations.append(new_raw.times[f], 0, "F_peak")
            for p in p_peaks:
                new_raw.annotations.append(new_raw.times[p], 0, "P_peak")


            # # now find streches with very low theta power for noise cov
            # max_pow = pow_dat.max(axis=0) # max of F and P
            # under_inds = max_pow < 0.5
            # switch_inds = np.where(under_inds[:-1] != under_inds[1:])[0]
            # is_on = under_inds[0]
            # last_idx = 0
            # inds = []
            # for si in switch_inds:
            #     if is_on:
            #         if (si - last_idx) > int(raw.info["sfreq"] * 2):
            #             inds.append((last_idx, si))
            #     last_idx = si
            #     is_on = not is_on
            #
            # # crop out
            # raws = []
            # for ind in inds:
            #     raws.append(raw.copy().crop(tmin=raw.times[ind[0]],
            #                                 tmax=raw.times[ind[1]]))
            # cov_raw = raws[0]
            # cov_raw.append(raws[1:])
            # cov = mne.compute_raw_covariance(cov_raw.copy().apply_proj())
            # cov.save(join(sess_dir, f"MT-YG-{subj}_{pp}{sess}-cov.fif"),
            #          overwrite=True)

            events = mne.events_from_annotations(new_raw)
            epo = mne.Epochs(raw, *events, tmin=-0.5, tmax=0.5, baseline=None,
                             event_repeated="merge")
            epo.save(join(sess_dir, f"MT-YG-{subj}_{pp}{sess}-epo.fif"),
                     overwrite=True)