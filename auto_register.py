from os.path import join
from os import listdir
import mne
from mne.coreg import Coregistration
import pandas as pd
import re
import numpy as np

root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
data_dir = join(root_dir, mem_dir, "02_MemTask")
pos_dir = join(root_dir, mem_dir, "01_BrainSight")
subjects_dir = join(root_dir, "hdd/freesurfer/subjects")

subjs = listdir(data_dir)
preposts = ["pre", "post"]

for subj in subjs:
    match = re.match("MT-YG-(\d{3})", subj)
    if not match:
        continue
    subj_dir = join(data_dir, subj)
    for sess in listdir(subj_dir):
        if not re.match("Session\d", sess):
            continue
        sess_dir = join(subj_dir, sess, "EEG")
        raw = mne.io.Raw(join(sess_dir, f"{subj}_{sess}_pre-raw.fif"),
                         preload=True)
        # get planned fiducials
        fid_df = pd.read_csv(join(pos_dir, f"{subj}_{sess}_planned.txt"),
                             delimiter="\t")
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

        plot_kwargs = dict(subject=subj, subjects_dir=subjects_dir,
               surfaces="head-dense", dig=True, show_axes=True, mri_fiducials=True)
        coreg = Coregistration(raw.info, subj, fiducials="auto", subjects_dir=subjects_dir)
        #fig = mne.viz.plot_alignment(raw.info, trans=coreg.trans, **plot_kwargs)
        coreg.fit_fiducials()
        #fig = mne.viz.plot_alignment(raw.info, trans=coreg.trans, **plot_kwargs)
        coreg.fit_icp(n_iterations=100, nasion_weight=0, lpa_weight=0, rpa_weight=0, verbose=True)
        fig = mne.viz.plot_alignment(raw.info, trans=coreg.trans, **plot_kwargs)
        print(f"\n{subj} {sess}\n")
        breakpoint()
        mne.write_trans(join(sess_dir, f'{subj}_{sess}_auto-trans.fif'),
                        coreg.trans)
        fig.plotter.close()
