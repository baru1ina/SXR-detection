import numpy as np

def detect_crashes(score, threshold_factor=3):
    mean = np.mean(score)
    std = np.std(score)
    threshold = mean + threshold_factor * std
    indices = np.where(score > threshold)[0]
    return indices
