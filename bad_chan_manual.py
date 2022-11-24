import mne
from os.path import join

root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
data_dir = join(root_dir, mem_dir, "02_MemTask")

recs = [[120, 1, "post", ["FT9"]], [125, 1, "post", ["PO4"]],
        [132, 2, "pre", ["AF7"]], [133, 2, "pre", ["FT7", "FCz"]],
        [121, 2, "pre", ["AF3", "TP9"]],
        [121, 2, "post", ["AF3", "TP10", "TP9"]],
        [144, 1, "post", ["C1", "TP10"]], [142, 1, "post", ["FC1"]]]

for rec in recs:
    (subj, sess, pp, bad_chans) = rec
    raw = mne.io.Raw(join(data_dir, f"MT-YG-{subj}", f"Session{sess}", "EEG",
                          f"MT-YG-{subj}_Session{sess}_{pp}-raw.fif"),
                     preload=True)
    if len(bad_chans):
        raw.info["bads"].extend(bad_chans)
    else:
        raw.plot(n_channels=60, duration=120)
    breakpoint()
    raw.save(join(data_dir, f"MT-YG-{subj}", f"Session{sess}", "EEG",
                  f"MT-YG-{subj}_Session{sess}_{pp}-raw.fif"), overwrite=True)
