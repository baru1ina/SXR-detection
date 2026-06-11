import numpy as np

def robust_scale(signal: np.ndarray):
    med = np.median(signal, axis=0)
    mad = np.median(np.abs(signal - med), axis=0) + 1e-8
    return (signal - med) / mad
