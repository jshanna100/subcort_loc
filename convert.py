import mne
from os.path import join
from mne.io import read_raw_brainvision
import pandas as pd
import numpy as np
from anoar import BadChannelFind

def file_to_montage(pos_dir, subj, sess):
    # read brainsight files and output as MNE montage
    elec_df = pd.read_csv(join(pos_dir, f"MT-YG-{subj}_Session{sess}_eeg.txt"),
                          delimiter="\t")
    fid_df = pd.read_csv(join(pos_dir, f"MT-YG-{subj}_Session{sess}_planned.txt"),
                         delimiter="\t")
    elec_dict = {}
    for idx, row in elec_df.iterrows():
        elec_dict[row["Electrode Name"]] = np.array([row["Loc. X"],
                                                     row["Loc. Y"],
                                                     row["Loc. Z"]])
        elec_dict[row["Electrode Name"]] *= 1e-3
    for idx, row in fid_df.iterrows():
        label = row["Planned Landmark Name"]
        pos = np.array([row["Loc. X"], row["Loc. Y"], row["Loc. Z"]])
        pos *= 1e-3
        if label == "Links":
            lpa = pos
        elif label == "Rechts":
            rpa = pos
        elif label == "Nasion":
            nasion = pos
        elif label == "Nasenspitze":
            hsp = pos[None, ]

    digmon = mne.channels.make_dig_montage(nasion=nasion, lpa=lpa, rpa=rpa,
                                           ch_pos=elec_dict, hsp=hsp)
    return digmon


root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
data_dir = join(root_dir, mem_dir, "02_MemTask")
pos_dir = join(root_dir, mem_dir, "01_BrainSight")

subjs = ["120"]
sessions = ["2"]
preposts = ["pre", "post"]

for subj in subjs:
    for sess in sessions:
        subj_dir = join(data_dir, f"MT-YG-{subj}", f"Session{sess}", "EEG")
        mont = file_to_montage(pos_dir, subj, sess)
        for pp in preposts:
            raw = read_raw_brainvision(join(subj_dir,
                                            f"MT-YG-{subj}_{pp}{sess}.vhdr"))
            raw.resample(200)
            raw.filter(l_freq=0.3, h_freq=30)
            raw.set_channel_types({"VEOGu":"eog", "HEOGr":"eog"})
            raw.set_montage(mont)

            # bad channels
            eeg_picks = mne.pick_types(raw.info, eeg=True)
            bcf = BadChannelFind(eeg_picks, thresh=0.7)
            bads = bcf.recommend(raw)

            # make VEOG and HEOG channels
            eog_picks = mne.pick_channels(raw.ch_names,
                                          ["VEOGu", "HEOGr", "Fp1", "AF7"])
            eog_data = raw.get_data(eog_picks)
            new_eog = np.stack([eog_data[0,] - eog_data[2],
                                eog_data[1,] - eog_data[3]])
            info = mne.create_info(["VEOG", "HEOG"], raw.info["sfreq"],
                                   ch_types="eog")
            eog_raw = mne.io.RawArray(new_eog, info)
            raw.add_channels([eog_raw], force_update_info=True)
            raw.drop_channels(["VEOGu", "HEOGr"])
            raw.save(join(subj_dir, f"MT-YG-{subj}_{pp}{sess}-raw.fif"),
                     overwrite=True)
