import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter


def smooth_score(score, window=5):
    kernel = np.ones(window) / window
    return np.convolve(score, kernel, mode="same")


def smooth_signal_savgol(smoothing_window, smoothing_poly, signal):
    if smoothing_window >= len(signal):
        return signal
    if smoothing_window % 2 == 0:
        smoothing_window += 1
    return savgol_filter(signal, smoothing_window, smoothing_poly)


def smooth_signal_median(signal, base_window=5):
    smoothed = np.zeros_like(signal)
    for i in range(len(signal)):
        deriv = np.abs(np.gradient(signal))[i]
        window = base_window + int(deriv * 1000)
        window = min(window, 51) if window % 2 else window + 1
        left = max(0, i - window // 2)
        right = min(len(signal), i + window // 2 + 1)
        smoothed[i] = np.median(signal[left:right])
    return smoothed


def smooth_signal_gauss(signal, sigma=1.5, truncate=4.0):
    radius = int(truncate * sigma + 0.5)
    window_size = 2 * radius + 1
    kernel = np.zeros(window_size)
    center = radius
    for x in range(window_size):
        dx = x - center
        kernel[x] = np.exp(-(dx ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)
    smoothed = np.convolve(signal, kernel, mode='same')
    return smoothed