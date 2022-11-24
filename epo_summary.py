import mne
from os.path import join
from os import listdir
import matplotlib.pyplot as plt
import re
plt.ion()

root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
data_dir = join(root_dir, mem_dir, "02_MemTask")

subjs = listdir(data_dir)
preposts = ["pre", "post"]

fig, axes = plt.subplots(9, 10, figsize=(38.4, 21.6))
axes = axes.ravel()
ax_idx = 0
for subj in subjs:
    match = re.match("MT-YG-(\d{3})", subj)
    if not match:
        continue
    subj_dir = join(data_dir, subj)
    for sess in listdir(subj_dir):
        if not re.match("Session\d", sess):
            continue
        sess_dir = join(subj_dir, sess, "EEG")
        for pp in preposts:
            filename = f"{subj}_{sess}_{pp}-epo.fif"
            if filename in listdir(sess_dir):
                epo = mne.read_epochs(join(sess_dir, filename))
                epo.average().plot(axes=axes[ax_idx])
            axes[ax_idx].set_title(f"{subj}_{sess}_{pp}")
            ax_idx += 1
