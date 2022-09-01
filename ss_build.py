import mne

root_dir = "/home/jev/"
proc_dir = root_dir + "subcort_loc/proc/"
subjects_dir = root_dir + "hdd/freesurfer/subjects"
spacing = "ico5"
sc_names = ["Left-Hippocampus", "Left-Amygdala", "Right-Hippocampus",
            "Right-Amygdala"]
subjs = ["91", "92", "93"]

for subj in subjs:
    subj_str = "{}".format(subj)
    # cortex
    bem_model = mne.make_bem_model(subj_str, subjects_dir=subjects_dir, ico=5)
    bem = mne.make_bem_solution(bem_model)

    src = mne.setup_source_space(subj_str, subjects_dir=subjects_dir,
                                 n_jobs=16, spacing=spacing)
    # sub cortical
    vol_src = mne.setup_volume_source_space(subj_str, bem=bem,
                                            volume_label=sc_names,
                                            subjects_dir=subjects_dir)
    mix_src = src + vol_src
    mix_src.save("{}{}_{}-src.fif".format(proc_dir, subj, spacing))
    for blocks in ["AB", "CD"]:
        raw = mne.io.Raw("{}{}_{}_64-raw.fif".format(proc_dir, subj, blocks[0]))
        trans = "{}{}_{}-trans.fif".format(proc_dir, subj, blocks)
        fwd = mne.make_forward_solution(raw.info, trans, mix_src, bem, n_jobs=16)
        mne.write_forward_solution("{}{}_{}_{}-fwd.fif".format(proc_dir, subj,
                                                               blocks, spacing),
                                   fwd)
