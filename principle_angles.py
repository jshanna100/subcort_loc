import numpy as np
from itertools import product
from scipy.linalg import subspace_angles

def fwd_sa(a, b):
    angles = np.ones((a.shape[1], b.shape[1])) * np.nan
    combs = list(product(np.arange(a.shape[1]),
                         np.arange(b.shape[1])))
    for comb in combs:
        angle = np.rad2deg(subspace_angles(a[:, comb[0]][:, None],
                                           b[:, comb[1]][:, None])[0])
        angles[comb[0], comb[1]] = angle
    return angles
