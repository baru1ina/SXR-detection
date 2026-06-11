import numpy as np


def detect_plasma_interval(ip_signal, time, threshold_ratio=0.2):
    max_ip = np.max(ip_signal)

    if max_ip <= 0:
        return None

    threshold = threshold_ratio * max_ip

    mask = ip_signal > threshold

    if not np.any(mask):
        return None

    indices = np.where(mask)[0]

    start_idx = indices[0]
    end_idx = indices[-1]

    return time[start_idx], time[end_idx]