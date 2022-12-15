import mne
import numpy as np
from os.path import join
from os import listdir
import re
from cnx_utils import (TriuSparse, load_sparse, plot_directed_cnx,
                       plot_undirected_cnx, make_brain_image)
import matplotlib.pyplot as plt
plt.ion()

root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
fig_dir = join(mem_dir, "figures")
data_dir = join(root_dir, mem_dir, "02_MemTask")
subjects_dir = root_dir + "hdd/freesurfer/subjects"

inv_method="MNE"
snr = 1.0
lambda2 = 1.0 / snr ** 2

parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage", parc)

preposts = ["pre", "post"]
views = {"left":{"view":"lateral", "distance":625, "hemi":"lh"},
         "right":{"view":"lateral", "distance":625, "hemi":"rh"},
         "upper":{"view":"dorsal", "distance":625,
                  "focalpoint":(-.77, 3.88, -21.53)},
         "caudal":{"view":"caudal", "distance":625}
        }
stim_key = {"MT-YG-120":{"Session1":"active", "Session2":"sham"},
            "MT-YG-121":{"Session1":"active", "Session2":"sham"},
            "MT-YG-128":{"Session1":"sham", "Session2":"active"},
            "MT-YG-127":{"Session1":"sham", "Session2":"active"},
            "MT-YG-138":{"Session1":"sham", "Session2":"active"},
            "MT-YG-133":{"Session1":"active", "Session2":"sham"},
            "MT-YG-136":{"Session1":"active", "Session2":"sham"},
            "MT-YG-132":{"Session1":"sham", "Session2":"active"},
            "MT-YG-134":{"Session1":"active", "Session2":"sham"},
            "MT-YG-125":{"Session1":"sham", "Session2":"active"},
            "MT-YG-131":{"Session1":"sham", "Session2":"active"},
            "MT-YG-122":{"Session1":"active", "Session2":"sham"},
            "MT-YG-124":{"Session1":"sham", "Session2":"active"},
            "MT-YG-139":{"Session1":"active", "Session2":"sham"},
            "MT-YG-141":{"Session1":"active", "Session2":"sham"},
            "MT-YG-140":{"Session1":"sham", "Session2":"active"},
            "MT-YG-148":{"Session1":"sham", "Session2":"active"},
            "MT-YG-144":{"Session1":"sham", "Session2":"active"},
            "MT-YG-146":{"Session1":"active", "Session2":"sham"},
            "MT-YG-142":{"Session1":"active", "Session2":"sham"},
            "MT-YG-137":{"Session1":"active", "Session2":"sham"},
            "MT-YG-147":{"Session1":"sham", "Session2":"active"}
            }

exclu = ["MT-YG-124"]

all_stc = {"active_pre":None, "active_post":None, "sham_pre":None, "sham_post":None}
all_stc_n = {"active_pre":0, "active_post":0, "sham_pre":0, "sham_post":0}
subjs = listdir(data_dir)
for subj in subjs:
    if subj in exclu:
        continue
    match = re.match("MT-YG-(\d{3})", subj)
    if not match:
        continue
    subj_dir = join(data_dir, subj)
    for sess in listdir(subj_dir):
        if not re.match("Session\d", sess):
            continue
        sess_dir = join(subj_dir, sess, "EEG")
        cond = stim_key[subj][sess]
        for pp_idx, pp in enumerate(preposts):
            try:
                stc = mne.read_source_estimate(join(sess_dir,
                                                    f"{subj}_{sess}_{pp}"))
            except:
                continue
            if all_stc[f"{cond}_{pp}"] is None:
                all_stc[f"{cond}_{pp}"] = stc
            else:
                all_stc[f"{cond}_{pp}"] += stc
            all_stc_n[f"{cond}_{pp}"] += 1


lims = [.2e-13, .7e-13, 1.2e-13]
clim = {"kind":"value", "lims":lims}
#clim = "auto"
fig, axes = plt.subplots(4, 1, figsize=(21.6, 21.6))
for idx, (k, v) in enumerate(all_stc.items()):
    v /= all_stc_n[k]
    brain = v.plot(subject="fsaverage", hemi="both",
                   clim=clim)
    img = make_brain_image(views, brain, text=f"{k}", text_pan=0)
    axes[idx].imshow(img)
    axes[idx].axis("off")
plt.tight_layout()
fig.savefig(join(fig_dir, "grand_ctx_amp.png"))
