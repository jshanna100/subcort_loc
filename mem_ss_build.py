import mne
from os.path import join
from os import listdir
import numpy as np

'''
Build source spaces and bems
'''

root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
data_dir = join(root_dir, mem_dir, "02_MemTask")
data_list = listdir(data_dir)

subjs = [x for x in np.arange(120,150)]

n_jobs = 16
subjects_dir = root_dir + "hdd/freesurfer/subjects"
spacing = "ico5"
sc_base = ["Caudate", "Putamen", "Hippocampus", "Amygdala"]
sc_names = [f"Left-{x}" for x in sc_base] +  [f"Right-{x}" for x in sc_base]

excludes = [135]

for subj in subjs:
    if subj in excludes:
        continue
    subj_str = f"MT-YG-{subj}"
    if subj_str not in data_list:
        print(f"{subj_str} not found. Skipping.")
        continue
    subj_dir = join(data_dir, subj_str)

    try:
        mne.bem.make_watershed_bem(subj_str, subjects_dir=subjects_dir)
    except:
        print("BEM already exists.")
    try:
        mne.bem.make_scalp_surfaces(subj_str, subjects_dir=subjects_dir)
    except:
        print("Scalp surfaces already exist.")

    bem_filename = f"MT-YG-{subj}-bem.fif"
    if bem_filename not in listdir(subj_dir):
        bem_model = mne.make_bem_model(subj_str, subjects_dir=subjects_dir,
                                       ico=4)
        bem = mne.make_bem_solution(bem_model)
        mne.write_bem_solution(join(subj_dir, f"MT-YG-{subj}-bem.fif"), bem)
    else:
        bem = mne.read_bem_solution(join(subj_dir, f"MT-YG-{subj}-bem.fif"))
    src_filename = f"MT-YG-{subj}-src.fif"
    if src_filename not in listdir(subj_dir):
        src = mne.setup_source_space(subj_str, subjects_dir=subjects_dir,
                                     n_jobs=n_jobs, spacing=spacing)
        src.save(join(subj_dir, src_filename), overwrite=True)
        # sub cortical
        vol_src = mne.setup_volume_source_space(subj_str, bem=bem,
                                                volume_label=sc_names,
                                                subjects_dir=subjects_dir)

        vol_src.save(join(subj_dir, f"MT-YG-{subj}_vol-src.fif"), overwrite=True)
