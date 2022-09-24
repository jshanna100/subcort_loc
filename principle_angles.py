def principle_angle(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# principle angles
## cortex to cortex
gain = mix_fwd["sol"]["data"]
gain = np.stack((gain[:, 0::3], gain[:, 1::3], gain[:, 2::3]))
gain = np.linalg.norm(gain, axis=0)
ctx_inds = mix_fwd["src"][0]["nuse"] + mix_fwd["src"][1]["nuse"]
sub_inds = ctx_inds + sum([s["nuse"] for s in mix_fwd["src"][2:]])
sub_n = sub_inds - ctx_inds
rand_ctx_inds = np.random.randint(0, ctx_inds, sub_n)
ctx_combs = list(combinations(rand_ctx_inds, 2))
ctx_angles = np.zeros(len(ctx_combs))
for idx, c_comb in enumerate(ctx_combs):
    ctx_angles[idx] = principle_angle(gain[:, c_comb[0]],
                                      gain[:, c_comb[1]])
plt.hist(ctx_angles, bins=100, alpha=0.3)
## subcortical to subcortical
sub_combs = list(combinations(np.arange(ctx_inds, sub_inds), 2))
sub_angles = np.zeros(len(sub_combs))
for idx, s_comb in enumerate(sub_combs):
    sub_angles[idx] = principle_angle(gain[:, s_comb[0]],
                                      gain[:, s_comb[1]])
plt.hist(sub_angles, bins=100, alpha=0.3)
## subcortical to cortical
sc_combs = list(product(rand_ctx_inds,
                        np.arange(ctx_inds, sub_inds)))
sc_combs = [sc_combs[i]
            for i in np.random.randint(0, len(sc_combs),
                                       len(sub_combs))]
sc_angles = np.zeros(len(sc_combs))
for idx, sc_comb in enumerate(sc_combs):
    sc_angles[idx] = principle_angle(gain[:, sc_comb[0]],
                                     gain[:, sc_comb[1]])
plt.hist(sc_angles, bins=100, alpha=0.3)
