import numpy as np
from itertools import product

def principle_angle(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def principle_angle_mat(m1, m2):
    m1 = m1 / np.linalg.norm(m1, axis=0)
    m2 = m2 / np.linalg.norm(m2, axis=0)
    angle_mat = np.arccos(np.clip(np.matmul(m1.T, m2), -1., 1.))
    return angle_mat


def gain_angle_on_gain(gain_to, gain_from):
    pangle_mat = principle_angle_mat(gain_to, gain_from)
    return pangle_mat
