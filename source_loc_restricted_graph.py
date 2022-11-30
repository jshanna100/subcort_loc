import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from os.path import join

root_dir = "/home/jev/"
mem_dir = join(root_dir, "hdd", "memtacs", "pilot")
fig_dir = join(mem_dir, "figures")
data_dir = join(root_dir, mem_dir, "02_MemTask")


signal_df = pd.read_pickle(join(fig_dir, "signal.pickle"))
cnx_df = pd.read_pickle(join(fig_dir, "cnx.pickle"))

excludes = [124]
for excl in excludes:
    signal_df = signal_df.query(f"subj != 'MT-YG-{excl}'")
    cnx_df = cnx_df.query(f"subj != 'MT-YG-{excl}'")

subjs = list(cnx_df["subj"].unique())
reg_names = list(cnx_df["from_reg"].unique())

# grand averages
# get dataframes ready
# signal
abs_df = signal_df.copy()
abs_df["amp"] = abs_df["amp"].abs()
grand_signal = abs_df.groupby(["subj", "stim", "pp", "reg", "time"],
                              as_index=False).mean()
# cnx
grand_cnx = cnx_df.groupby(["stim", "pp", "to_reg", "from_reg"],
                           as_index=False).mean()

fig, axes = plt.subplots(4, 4, figsize=(38.4, 21.6))
idx = 0
# get overall subj max and min
max_wpli = grand_cnx["wpli"].max()
min_dpli, max_dpli = grand_cnx["dpli"].min(), grand_cnx["dpli"].max()
min_dpte, max_dpte = grand_cnx["dpte"].min(), grand_cnx["dpte"].max()
for stim in ["active", "sham"]:
    stim_cnx = grand_cnx.query(f"stim=='{stim}'")
    stim_signal = grand_signal.query(f"stim=='{stim}'")
    for pp in ["pre", "post"]:
        pp_cnx = stim_cnx.query(f"pp=='{pp}'")
        pp_signal = stim_signal.query(f"pp=='{pp}'")
        sns.lineplot(data=pp_signal, x="time", y="amp", hue="reg",
                     hue_order=reg_names, ax=axes[idx, 0])
        #axes[idx, 0].set_ylim(1e-9, 2.5e-9)
        axes[idx, 0].set_ylim(2e-9, 7e-9)
        axes[idx, 0].set_title(f"{stim}, {pp}", fontweight="bold")
        wpli = pp_cnx.pivot("from_reg", "to_reg", "wpli")
        sns.heatmap(data=wpli, cmap="inferno", vmin=0, vmax=max_wpli,
                    ax=axes[idx, 1], annot=True, fmt=".2f")
        dpli = pp_cnx.pivot("from_reg", "to_reg", "dpli")
        sns.heatmap(data=dpli, cmap="seismic", vmin=min_dpli, vmax=max_dpli,
                    ax=axes[idx, 2], center=.5, annot=True, fmt=".3f")
        dpte = pp_cnx.pivot("from_reg", "to_reg", "dpte")
        sns.heatmap(data=dpte, cmap="seismic", vmin=min_dpte, vmax=max_dpte,
                    ax=axes[idx, 3], center=.5, annot=True, fmt=".3f")
        idx += 1

axes[0, 1].set_title("wPLI", fontweight="bold")
axes[0, 2].set_title("dPLI", fontweight="bold")
axes[0, 3].set_title("dPTE", fontweight="bold")
plt.suptitle("grand", fontsize=24, fontweight="bold")
plt.tight_layout()
fig.savefig(join(fig_dir, "grand.png"))

for subj in subjs:
    fig, axes = plt.subplots(4, 4, figsize=(38.4, 21.6))
    idx = 0
    # get overall subj max and min
    subj_cnx = cnx_df.query(f"subj=='{subj}'")
    subj_signal = signal_df.query(f"subj=='{subj}'")
    max_wpli = subj_cnx["wpli"].max()
    min_dpli, max_dpli = subj_cnx["dpli"].min(), subj_cnx["dpli"].max()
    min_dpte, max_dpte = subj_cnx["dpte"].min(), subj_cnx["dpte"].max()
    for stim in ["active", "sham"]:
        stim_cnx = subj_cnx.query(f"stim=='{stim}'")
        stim_signal = subj_signal.query(f"stim=='{stim}'")
        for pp in ["pre", "post"]:
            pp_cnx = stim_cnx.query(f"pp=='{pp}'")
            pp_signal = stim_signal.query(f"pp=='{pp}'")
            sns.lineplot(data=pp_signal, x="time", y="amp", hue="reg",
                         hue_order=reg_names, ax=axes[idx, 0])
            axes[idx, 0].set_ylim(-4e-9, 4e-9)
            axes[idx, 0].set_title(f"{stim}, {pp}", fontweight="bold")
            wpli = pp_cnx.pivot("from_reg", "to_reg", "wpli")
            sns.heatmap(data=wpli, cmap="inferno", vmin=0, vmax=max_wpli,
                        ax=axes[idx, 1], annot=True, fmt=".2f")
            dpli = pp_cnx.pivot("from_reg", "to_reg", "dpli")
            sns.heatmap(data=dpli, cmap="seismic", vmin=min_dpli, vmax=max_dpli,
                        ax=axes[idx, 2], center=.5, annot=True, fmt=".3f")
            dpte = pp_cnx.pivot("from_reg", "to_reg", "dpte")
            sns.heatmap(data=dpte, cmap="seismic", vmin=min_dpte, vmax=max_dpte,
                        ax=axes[idx, 3], center=.5, annot=True, fmt=".3f")
            idx += 1

    axes[0, 1].set_title("wPLI", fontweight="bold")
    axes[0, 2].set_title("dPLI", fontweight="bold")
    axes[0, 3].set_title("dPTE", fontweight="bold")
    plt.suptitle(subj, fontsize=24, fontweight="bold")
    plt.tight_layout()
    fig.savefig(join(fig_dir, f"{subj}.png"))
    plt.close("all")
