import numpy as np

def create_windows(signals, window_size, stride=None):
    if stride is None:
        stride = max(window_size // 4, 1)
    X, Y = [], []
    for i in range(0, len(signals) - window_size - 1, stride):
        X.append(signals[i:i+window_size])
        Y.append(signals[i+window_size])

    return np.array(X), np.array(Y)


def estimate_window_size(shot, approx_period_ms=2.0):
    dt = shot.dt 
    period_sec = approx_period_ms * 1e-3
    samples_per_period = int(period_sec / dt)
    window_size = int(samples_per_period * 0.5)

    return max(window_size, 10)