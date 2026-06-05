import numpy as np
from src.preprocessing.windowing import create_windows
from src.preprocessing.derivative import compute_derivative
from src.preprocessing.normalization import robust_scale
from src.preprocessing.cleaning import remove_mean


def build_dataset(shots, window_size):

    X_all = []
    Y_all = []

    for shot in shots:

        signals = remove_mean(shot.signals)
        signals = robust_scale(signals)

        deriv = compute_derivative(signals, shot.dt)
        deriv = robust_scale(deriv)

        X, Y = create_windows(deriv, window_size)

        X_all.append(X)
        Y_all.append(Y)

    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)

    return X_all, Y_all