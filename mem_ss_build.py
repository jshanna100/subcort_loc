import mne
from os.path import join

root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
data_dir = join(root_dir, mem_dir, "02_MemTask")

subjs = ["120"]
sessions = ["2"]
preposts = ["pre", "post"]

subjects_dir = root_dir + "/freesurfer/subjects"
spacing = "ico5"
sc_base = ["Caudate", "Putamen", "Hippocampus", "Amygdala"]
sc_names = [f"Left-{x}" for x in sc_base] +  [f"Right-{x}" for x in sc_base]

for subj in subjs:
    subj_str = f"MT-YG-{subj}"
    subj_dir = join(data_dir, subj_str)
    # cortex
    bem_model = mne.make_bem_model(subj_str, subjects_dir=subjects_dir, ico=4)
    bem = mne.make_bem_solution(bem_model)

    src = mne.setup_source_space(subj_str, subjects_dir=subjects_dir,
                                 n_jobs=16, spacing=spacing)
    # sub cortical
    vol_src = mne.setup_volume_source_space(subj_str, bem=bem,
                                            volume_label=sc_names,
                                            subjects_dir=subjects_dir)
    mix_src = src + vol_src
    mix_src.save(join(subj_dir, f"MT-YG-{subj}-src.fif"), overwrite=True)
    for sess in sessions:
        sess_dir = join(subj_dir, f"Session{sess}", "EEG")
        trans = join(sess_dir, f"MT-YG-{subj}_Session{sess}-trans.fif")
        raw = mne.io.Raw(join(sess_dir, f"MT-YG-{subj}_pre{sess}-raw.fif"))
        mix_fwd = mne.make_forward_solution(raw.info, trans, mix_src, bem, n_jobs=16)
        mne.write_forward_solution(join(sess_dir,
                                        f"MT-YG-{subj}_Session{sess}-mix-fwd.fif"),
                                   mix_fwd, overwrite=True)
        ctx_fwd = mne.make_forward_solution(raw.info, trans, src, bem, n_jobs=16)
        mne.write_forward_solution(join(sess_dir,
                                        f"MT-YG-{subj}_Session{sess}-ctx-fwd.fif"),
                                   ctx_fwd, overwrite=True)
