import numpy as np

def remove_mean(signal: np.ndarray):
    return signal - np.mean(signal, axis=0)

def simple_detrend(signal: np.ndarray):
    return remove_mean(signal)
