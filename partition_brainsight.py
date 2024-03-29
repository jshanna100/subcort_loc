from os import listdir
from os.path import join
import re

"""
Fix the brainsight files so they can be more easily read by MNE-Python
"""


root_dir = "/home/jev/"
data_dir = join(root_dir, "hdd/memtacs/exp/01_BrainSight")
filenames = listdir(data_dir)

excludes = []

for filename in filenames:
    match = re.match("MT-OG-(\d{3})_Session(\d).txt", filename)
    if not match:
        continue
    (subj, sess) = match.groups()
    if subj in excludes:
        continue
    with open(join(data_dir, filename), "rt") as f:
        lines = f.readlines()

    starts = {}
    for idx, line in enumerate(lines):
        if "# Electrode Name" in line:
            starts["eeg"] = idx
        elif "# Planned Landmark" in line:
            starts["planned"] = idx
        elif "# Session Landmark" in line:
            starts["session"] = idx
        elif "# Session Name" in line:
            starts["session_name"] = idx
    line_inds = {"eeg":(starts["eeg"], starts["planned"]),
                 "planned":(starts["planned"], starts["session"]),
                 "session":(starts["session"], starts["session_name"])
                }

    for v in starts.values():
        lines[v] = lines[v][2:] # Get rid of "# " at beginning

    for k, v in line_inds.items():
        with open(join(data_dir, f"MT-OG-{subj}_Session{sess}_{k}.txt"), "wt") as f:
            f.writelines(lines[v[0]:v[1]])
