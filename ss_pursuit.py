import mne
from os.path import join
from os import listdir
from mne.inverse_sparse.subspace_pursuit import (make_patch_forward,
                                                 subspace_pursuit,
                                                 telescope_vertices)
import numpy as np

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
    bem_file = f"MT-YG-{subj}-ctx-bem.fif"
    if bem_file in listdir(subj_dir):
        bem = mne.read_bem_solution(join(subj_dir, bem_file))
    else:
        bem_model = mne.make_bem_model(subj_str, subjects_dir=subjects_dir,
                                       ico=4)
        bem = mne.make_bem_solution(bem_model)
        mne.write_bem_solution(join(subj_dir, bem_file), bem)

    for sess in sessions:
        sess_dir = join(subj_dir, f"Session{sess}", "EEG")
        trans = join(sess_dir, f"MT-YG-{subj}_Session{sess}-trans.fif")
        epo = mne.read_epochs(join(sess_dir, f"MT-YG-{subj}_pre{sess}-epo.fif"),
                              preload=True)
        epo = epo["P_peak"]
        epo.crop(tmin=-.25, tmax=.25)
        cov = mne.compute_covariance(epo, keep_sample_mean=False)
        evo = epo.average()
        s1_fwd_file = f"MT-YG-{subj}_Session{sess}_p_ico1-fwd.fif"
        if s1_fwd_file in listdir(sess_dir):
            s1_fwd = mne.read_forward_solution(join(sess_dir, s1_fwd_file))
        else:
            s1_fwd = make_patch_forward(subj_str, "ico1", bem, evo.info, trans,
                                        patch_comp_n=0.9, n_jobs=16)
            mne.write_forward_solution(join(sess_dir, s1_fwd_file), s1_fwd)

        ss1_stc, var_expl = subspace_pursuit(s1_fwd, evo, cov, 2, 0.5, 9,
                                            return_var_expl=True)
        new_verts = telescope_vertices(subj_str, "ico1", "ico2",
                                       ss1_stc.vertices, n_jobs=16)
        s2_fwd_file = f"MT-YG-{subj}_Session{sess}_p_ico2-fwd.fif"
        if s2_fwd_file in listdir(sess_dir):
            s2_fwd = mne.read_forward_solution(join(sess_dir, s2_fwd_file))
        else:
            s2_fwd = make_patch_forward(subj_str, "ico2", bem, evo.info, trans,
                                        restrict_to=new_verts, patch_comp_n=0.9,
                                        n_jobs=16)
            #mne.write_forward_solution(join(sess_dir, s2_fwd_file), s2_fwd)
        ss2_stc, var_expl = subspace_pursuit(s2_fwd, evo, cov, 4, 0.5, 9,
                                            return_var_expl=True)
        new_verts = telescope_vertices(subj_str, "ico2", "ico3",
                                       ss2_stc.vertices, n_jobs=16)
        s3_fwd_file = f"MT-YG-{subj}_Session{sess}_p_ico3-fwd.fif"
        if s3_fwd_file in listdir(sess_dir):
            s3_fwd = mne.read_forward_solution(join(sess_dir, s3_fwd_file))
        else:
            s3_fwd = make_patch_forward(subj_str, "ico3", bem, evo.info, trans,
                                        restrict_to=new_verts, patch_comp_n=0.9,
                                        n_jobs=16)
            #mne.write_forward_solution(join(sess_dir, s3_fwd_file), s3_fwd)
        ss3_stc, var_expl = subspace_pursuit(s3_fwd, evo, cov, 8, 0.5, 9,
                                            return_var_expl=True)
