import mne
from os.path import join
from os import listdir
import re
'''
Clean ocular artefacts with ICA
'''


root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
data_dir = join(root_dir, mem_dir, "02_MemTask")

subjs = listdir(data_dir)
preposts = ["pre", "post"]

recs = [[120, 1, "post", ["FT9"]], [125, 1, "post", ["PO4"]],
        [132, 2, "pre", ["AF7"]], [133, 2, "pre", ["FT7", "FCz"]],
        [121, 2, "pre", ["AF3", "TP9"]],
        [121, 2, "post", ["AF3", "TP10", "TP9"]],
        [144, 1, "post", ["C1", "TP10"]], [142, 1, "post", ["FC1"]]]

overwrite = True
incl = ["120", "121", "125", "132", "133", "142", "144"]

for subj in subjs:
    match = re.match("MT-YG-(\d{3})", subj)
    if not match:
        continue
    if match.groups()[0] not in incl:
        continue
    subj_dir = join(data_dir, subj)
    for sess in listdir(subj_dir):
        if not re.match("Session\d", sess):
            continue
        sess_dir = join(subj_dir, sess, "EEG")
        for pp in preposts:
            outfile = f"{subj}_{sess}_{pp}_ica-raw.fif"
            if outfile in listdir(sess_dir) and not overwrite:
                print("Already exists. Skipping...")
                continue
            raw = mne.io.Raw(join(sess_dir, f"{subj}_{sess}_{pp}-raw.fif"),
                             preload=True)
            ica = mne.preprocessing.ICA(method="picard")
            ica.fit(raw)
            eog_bads, _ = ica.find_bads_eog(raw)
            new_raw = ica.apply(raw, exclude=eog_bads)
            new_raw.save(join(sess_dir, outfile), overwrite=True)
