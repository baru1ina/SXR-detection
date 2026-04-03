import numpy as np

def crash_score_derivative(y_true, y_pred):
    error = np.mean((y_true - y_pred) ** 2, axis=1)
    negative_spike = np.minimum(y_true, 0)
    spike_strength = np.mean(np.abs(negative_spike), axis=1)

    return error + 2.0 * spike_strength