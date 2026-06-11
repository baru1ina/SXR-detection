import numpy as np
import matplotlib.pyplot as plt

from config.path import PATH_TO_RES_AUTOCORR
from matplotlib.widgets import CheckButtons

from src.physics.autocorr_period import sliding_autocorr_period, autocorr_period, build_period_map, robust_period_from_map, build_saw_activity_mask
from src.physics.fft import fft_score, fft_score_with_drift

from scipy.signal import butter, filtfilt

from src.anomaly.smoothing import smooth_signal_median
#
# def is_sawtooth(interval, shot, drop_threshold=2.0):
#     start, end = interval
#     segment = shot.signals[start:end]
#     diff = np.diff(segment, axis=0)
#     max_drop = np.min(diff)
#     return max_drop < -drop_threshold


def detect_sawtooth_by_crashes(signal, dt,
                               neg_threshold_sigma=3,
                               min_crashes=5):

    signal = signal - np.mean(signal)

    deriv = np.gradient(signal, dt)

    sigma = np.std(deriv)

    if sigma == 0:
        return False, None, None

    # сильные отрицательные скачки
    crash_indices = np.where(deriv < -neg_threshold_sigma * sigma)[0]

    if len(crash_indices) < min_crashes:
        return False, None, None

    # грубая оценка периода
    crash_times = crash_indices * dt
    intervals = np.diff(crash_times)

    if len(intervals) < 2:
        return False, None, None

    period = np.median(intervals)

    return True, (0, len(signal)), period


def highpass_filter(signal, dt, cutoff_freq=50):
    nyquist = 0.5 / dt
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype='high')
    return filtfilt(b, a, signal)


def detect_sawtooth_hybrid(
        signal,
        prehist,
        dt,
        logger,
        ch,
        ac_threshold=0.3,
        fft_threshold=0.1,
        min_fraction=0.5,
        min_snr=2):

    noise_estimate = np.std(prehist)
    signal_amplitude = np.std(signal)
    snr = signal_amplitude / (noise_estimate + 1e-8)
    logger.info(f"Signal-to-noise ratio = {snr:.2f}")

    if snr < min_snr:
        logger.warning(f"SNR too low: {snr:.2f} < {min_snr}")
        return False, None, None, None, None

    times, periods, scores = sliding_autocorr_period(
        signal,
        logger,
        dt
    )

    estimated_period = robust_period_from_map(periods, scores, ac_threshold=ac_threshold)

    logger.info(f"estimated_period = {estimated_period}")

    period_map = build_period_map(
        n_samples=len(signal),
        times=times,
        periods=periods,
        scores=scores,
        ac_threshold=ac_threshold,
        default_period=estimated_period,
        include_boundary_periods=True,
    )

    saw_mask = build_saw_activity_mask(
        n_samples=len(signal),
        times=times,
        periods=periods,
        scores=scores,
        dt=dt,
        global_period=estimated_period,
        ac_threshold=ac_threshold,
    )

    fft_ratio, peakiness, f0_array, is_saw = fft_score_with_drift(
        signal, logger, dt, times, periods, scores
    )

    if not is_saw:
        return False, None, None, None, None

    logger.info(f"fft_ratio = {fft_ratio}")

    interval = (0, len(signal))

    return True, interval, estimated_period, period_map, saw_mask