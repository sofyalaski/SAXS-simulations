import numpy as np


def compute_error(d1, d2):   
    return np.max(np.abs(d1-d2) / np.mean((np.abs(d1), np.abs(d2)), axis=0))