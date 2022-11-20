import mne
from os.path import join
from os import listdir
from mne.io import read_raw_brainvision
import pandas as pd
import numpy as np
from anoar import BadChannelFind
import re

def file_to_montage(pos_dir, subj, sess):
    # read brainsight files and output as MNE montage
    elec_df = pd.read_csv(join(pos_dir, f"{subj}_{sess}_eeg.txt"),
                          delimiter="\t")
    fid_df = pd.read_csv(join(pos_dir, f"{subj}_{sess}_planned.txt"),
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
        if label == "LPA":
            lpa = pos
        elif label == "RPA":
            rpa = pos
        elif label == "Nasion":
            nasion = pos
        elif label == "Nasenspitze":
            hsp = pos[None, ]

    digmon = mne.channels.make_dig_montage(nasion=nasion, lpa=lpa, rpa=rpa,
                                           ch_pos=elec_dict, hsp=hsp)
    return digmon

overwrite = True
root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
data_dir = join(root_dir, mem_dir, "02_MemTask")
pos_dir = join(root_dir, mem_dir, "01_BrainSight")

sessions = ["1", "2"]
preposts = ["pre", "post"]

subjs = listdir(data_dir)
success_files = []
sess_dict = {"Session1":"", "Session2":"2"}
for subj in subjs:
    match = re.match("MT-YG-(\d{3})", subj)
    if not match:
        continue
    subj_dir = join(data_dir, subj)
    for sess in listdir(subj_dir):
        if not re.match("Session\d", sess):
            continue
        sess_dir = join(subj_dir, sess, "EEG")
        try:
            mont = file_to_montage(pos_dir, subj, sess)
        except:
            continue
        for pp in preposts:
            try:
                raw = read_raw_brainvision(join(sess_dir,
                                                f"{subj}_{pp}{sess_dict[sess]}.vhdr"))
            except:
                raw = read_raw_brainvision(join(sess_dir,
                                                f"{subj}_{pp}{sess[-1]}.vhdr"))
            filename = f"{subj}_{sess}_{pp}-raw.fif"
            if filename in listdir(sess_dir) and not overwrite:
                print("Already exists. Skipping...")
                continue
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
            raw.save(join(sess_dir, filename),
                     overwrite=overwrite)
