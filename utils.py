import numpy as np
import matplotlib.pyplot as plt
from mne.source_space import (_set_source_space_vertices, setup_source_space,
                              setup_volume_source_space)
from mne import make_forward_solution, convert_forward_solution
from mne.utils.numerics import _PCA
from mne.io.pick import pick_types, channel_type, _picks_by_type
from mne.io.constants import FIFF

def _whiten_gain(gain, info):
    pre_whitener = np.empty([len(gain), 1])
    for _, picks_ in _picks_by_type(info, ref_meg=False, exclude=[]):
        pre_whitener[picks_] = np.std(gain[picks_])
    wh_gain = gain / pre_whitener
    return wh_gain, pre_whitener

def _svd_gain(gain, info, n_components, whiten=False):
    gain = gain.copy()
    if whiten:
        gain, pre_whitener = _whiten_gain(gain, info)
    pca = _PCA(n_components=n_components, whiten=whiten)
    u, s, v = pca._fit(gain)
    if whiten:
        svd_gain = u[:, :pca.n_components_] * np.sqrt(gain.shape[0] - 1)
        svd_gain = svd_gain * pre_whitener
    else:
        svd_gain = u[:, :pca.n_components_] * s[:pca.n_components_]
    return svd_gain, u, s, v, pca.n_components_

def locate_vertex(vtx, labels):
    for lab_idx, lab in enumerate(labels):
        if vtx in lab.vertices:
            return lab
    return None

def _convert_forward_patch(fwd, new_nns, new_rrs, new_gain, new_verts,
                           copy=True):
    fwd = fwd.copy() if copy else fwd
    # make sure order corresponds to increasing vertices
    start_inds = np.cumsum([0] + [len(nv) for nv in new_verts[:-1]])
    ordered_inds = np.concatenate([np.argsort(nv) + start_idx
                                   for nv, start_idx in zip(new_verts, start_inds)])
    new_verts = [np.sort(nv) for nv in new_verts]
    fwd["nsource"] = len(new_nns)
    fwd['source_nn'] = new_nns[ordered_inds]
    fwd['source_rr'] = new_rrs[ordered_inds]
    fwd["sol"]["data"] = new_gain[:, ordered_inds]
    fwd["sol"]["ncol"] = len(new_nns)
    fwd["source_ori"] = FIFF.FIFFV_MNE_FIXED_ORI
    fwd['surf_ori'] = True
    fwd["_orig_sol"] = fwd["sol"]["data"]
    fwd["_orig_source_ori"] = FIFF.FIFFV_MNE_FIXED_ORI
    if fwd["src"].kind == "volume":
        for s in fwd["src"]:
            s["type"] = "discrete"
    _set_source_space_vertices(fwd["src"], new_verts)
    return fwd

def make_brain_image(views, brain, orient="horizontal", text="",
                     text_loc=None, text_pan=None, fontsize=160,
                     legend=None, legend_pan=None, cbar=None,
                     vmin=None, vmax=None, cbar_label=""):
    img_list = []
    axis = 1 if orient=="horizontal" else 0
    for k,v in views.items():
        brain.show_view(**v)
        scr = brain.screenshot()
        img_list.append(scr)
    if text != "":
        img_txt_list = []
        brain.add_text(0, 0.8, text, text_loc, font_size=fontsize, color=(1,1,1))
        for k,v in views.items():
            brain.show_view(**v)
            scr = brain.screenshot()
            img_txt_list.append(scr)
        img_list[text_pan] = img_txt_list[text_pan]
    if legend:
        legend_list = []
        leg = brain._renderer.plotter.add_legend(legend, bcolor="w")
        for k,v in views.items():
            brain.show_view(**v)
            scr = brain.screenshot()
            legend_list.append(scr)
        img_list[legend_pan] = legend_list[legend_pan]
    if orient == "square": # only works for 2x2
        h, w, _ = img_list[0].shape
        img = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        img[:h, :w, ] = img_list[0]
        img[:h, w:, ] = img_list[1]
        img[h:, :w, ] = img_list[2]
        img[h:, w:, ] = img_list[3]
    else:
        img = np.concatenate(img_list, axis=axis)

    if cbar:
        norm = Normalize(vmin, vmax)
        scalmap = cm.ScalarMappable(norm, cbar)
        if orient == "horizontal":
            colbar_size = (img_list[-1].shape[0]*4, img_list[-1].shape[1]/6)
        else:
            colbar_size = (img_list[-1].shape[0]/6, img_list[-1].shape[1]*4)
        colbar_size = np.array(colbar_size) / 100
        fig, ax = plt.subplots(1,1, figsize=colbar_size)
        colbar = plt.colorbar(scalmap, cax=ax, orientation=orient)
        ax.tick_params(labelsize=48)
        if orient == "horizontal":
            ax.set_xlabel(cbar_label, fontsize=48)
        else:
            ax.set_ylabel(cbar_label, fontsize=48)
        fig.tight_layout()
        fig.canvas.draw()
        mat = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        mat = mat.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = np.concatenate((img, mat), axis=abs(axis-1))

    if legend:
        brain._renderer.plotter.remove_legend()

    return img

def make_restricted_forward(subject, labels, bem, info, trans,
                            subjects_dir=None, volume=False, volume_label=None,
                            vol_src_inds=None, patch_comp_n=.9, meg=True,
                            eeg=True, ignore_ref=False, mindist=0., n_jobs=1):
    if volume and not volume_label:
        raise ValueError("If doing a volume source space, volume labels must "
                         "also be specified; volume_label is empty.")
    if volume:
        src = setup_volume_source_space(subject, bem=bem,
                                        subjects_dir=subjects_dir,
                                        mindist=mindist,
                                        volume_label=volume_label)
    else:
        src = setup_source_space(subject,
                                 subjects_dir=subjects_dir,
                                 n_jobs=n_jobs, verbose="warning")

    # make a forward model for each label
    new_nns, new_rrs, new_gain, new_verts = [], [], [], [[] for s in src]

    for lab_idx, lab in enumerate(labels):
        # restrict source space to patch and calc forward
        print(f"Label {lab_idx} of {len(labels)}")
        l_src = src.copy()
        verts = [[] for s in src]
        if volume:
            src_idx = vol_src_inds[lab_idx]
        else:
            src_idx = 0 if lab.hemi == "lh" else 1
        verts[src_idx] = lab.vertices
        _set_source_space_vertices(l_src, verts)
        fwd = make_forward_solution(info, trans, l_src,
                                    bem, meg=meg, eeg=eeg,
                                    mindist=mindist,
                                    ignore_ref=ignore_ref,
                                    n_jobs=n_jobs,
                                    verbose="warning")
        if not volume:
            fwd = convert_forward_solution(fwd, force_fixed=True)

        # reduce gain matrix dimensionality to n best components
        # or components that explain n variance
        gain = fwd["sol"]["data"].copy()
        reduce_gain, u, s, v, n_comps = _svd_gain(gain, fwd["info"],
                                                  patch_comp_n,
                                                  whiten=True)
        # get prominence of sources in decomposition and use those to derive
        # locations and orientations for components
        nn_weights = v[:n_comps,:].T * s[:n_comps].T
        # reduce this to norm of 3 orientations for location weights
        if not volume:
            rr_weights = nn_weights
        else:
            rr_weights = np.stack([nn_weights[0::3], nn_weights[1::3],
                                   nn_weights[2::3]])
            rr_weights = np.linalg.norm(rr_weights, axis=0)
            nn_weights = nn_weights / sum(nn_weights) # rows sum to 1
            rr_weights = rr_weights / sum(rr_weights) # rows sum to 1
        nns = fwd["source_nn"]
        rrs = fwd["source_rr"]
        # get nn and rr weighted by SVD component contribution
        w_nns = np.dot(nn_weights.T, nns)
        w_rrs = np.dot(rr_weights.T, rrs)
        # find vertex nums which are closest to new_rrs
        for rr, nn in zip(w_rrs, w_nns):
            dists = np.linalg.norm((rrs - rr), axis=1)
            # don't allow repeats of vertices; breaks stuff downstream
            idx = 0
            sort_inds = np.argsort(dists)
            closest_idx = sort_inds[idx]
            while lab.vertices[closest_idx] in new_verts[src_idx]:
                idx += 1
                closest_idx = sort_inds[idx]
            new_verts[src_idx].append(lab.vertices[closest_idx])

            new_rrs.append(rrs[closest_idx])
            new_nns.append(nn)
        new_gain.append(reduce_gain)
        print(f"Patch constructed from {fwd['nsource']} of "
                    f"{sum([ls['nuse'] for ls in l_src])} sources, "
                    f"reduced to {n_comps} components.")

    new_nns = np.vstack(new_nns)
    new_rrs = np.vstack(new_rrs)
    new_gain = np.hstack(new_gain)
    new_fwd = _convert_forward_patch(fwd, new_nns, new_rrs, new_gain, new_verts)

    return new_fwd
