
def r_vector(rad):
    x_bar, y_bar = np.cos(rad).mean(), np.sin(rad).mean()
    r_mean = circmean(rad, low=-np.pi, high=np.pi)
    r_norm = np.linalg.norm((x_bar, y_bar))
    return r_mean, r_norm

def phase_diff(ph_a, ph_b):
    si = np.exp(1j * np.abs(ph_a - ph_b))
    return si

def plot_r(r_norms, r_means, quantiles=(0., .25, .50, .75, 1.)):
    fig = plt.figure()
    axes = [plt.subplot(221, projection="polar"),
            plt.subplot(222, projection="polar"),
            plt.subplot(223, projection="polar"),
            plt.subplot(224, projection="polar")]
    for ax_idx, quant_idx in enumerate(range(len(quantiles)-1)):
        r_inds = ((r_means > np.quantile(r_means, quantiles[quant_idx])) &
                  (r_means < np.quantile(r_means, quantiles[quant_idx+1])))
        these_rs = r_means[r_inds]
        circ_hist_norm(axes[ax_idx], these_rs)


mni_roi_coords = {"LFrontal":np.array([-50, 32, 12]),
                  "LParietal":np.array([-42, -64, 36]),
                  "RFrontal":np.array([50, 32, 12]),
                  "RParietal":np.array([42, -64, 36])}
roi_cols = {"LFrontal":"red", "LParietal":"green", "RFrontal":"blue",
            "RParietal":"cyan"}

mni_coords = []
for v_idx, v in enumerate(verts):
    hemi = "lh" if v_idx < len(stcs[0].lh_vertno) else "rh"

    brain.add_foci(v, coords_as_verts=True, color="black", hemi=hemi,
                   scale_factor=0.5)
    mni_coords.append(mne.vertex_to_mni(v, hemis.index(hemi),
                                        subj_str,
                                        subjects_dir=subjects_dir))
mni_coords = np.array(mni_coords)
# identify foci nearest to MNI coords
closest_inds = {}
for k,v in mni_roi_coords.items():
    dists = np.sqrt(np.sum((mni_coords - v)**2, axis=1))
    closest_idx = np.argmin(dists)
    closest_inds[k] = closest_idx
    hemi = "lh" if closest_idx < len(stcs[0].lh_vertno) else "rh"
    brain.add_foci(verts[closest_idx], coords_as_verts=True,
                   color=roi_cols[k], hemi=hemi)


vert_list = [s["vertno"] for s in mix_fwd["src"]]
l_list = np.sort(np.array([closest_inds["LFrontal"],
                          closest_inds["LParietal"]]))
r_list = np.sort(np.array([closest_inds["RFrontal"],
                          closest_inds["RParietal"]]))
lr_verts = np.concatenate(stcs[0].vertices)
l_list = [lr_verts[v] for v in l_list]
r_list = [lr_verts[v] for v in r_list]
#vert_list[0], vert_list[1] = l_list, r_list
