import mne
from os.path import join
'''
Clean ocular artefacts with ICA
'''


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
        sess_dir = join(subj_dir, f"Session{sess}", "EEG")
        for pp in preposts:
            raw = mne.io.Raw(join(sess_dir, f"MT-YG-{subj}_{pp}{sess}-raw.fif"),
                             preload=True)
            ica = mne.preprocessing.ICA(method="picard")
            ica.fit(raw)
            eog_bads, _ = ica.find_bads_eog(raw)
            new_raw = ica.apply(raw, exclude=eog_bads)
            new_raw.save(join(sess_dir, f"MT-YG-{subj}_{pp}{sess}_ica-raw.fif"))
