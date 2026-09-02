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
        fft_threshold=0.4,
        min_fraction=0.25,
        min_snr=2,
        signal_for_snr=None):

    snr_signal = signal if signal_for_snr is None else signal_for_snr
    noise_estimate = np.std(np.asarray(prehist, dtype=float))
    signal_amplitude = np.std(np.asarray(snr_signal, dtype=float))
    snr = signal_amplitude / (noise_estimate + 1e-8)
    logger.info(
        f"Signal-to-noise ratio = {snr:.2f} "
        f"(signal_std={signal_amplitude:.6e}, noise_std={noise_estimate:.6e})"
    )

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
        include_boundary_periods=False,
        support_radius_samples=int(round(0.5 * 20e-3 / dt)),
        max_interp_gap_samples=int(round(1.5 * 5e-3 / dt)),
        fill_unknown=False,
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
        signal,
        logger,
        dt,
        times,
        periods,
        scores,
        ac_threshold=ac_threshold,
        harmonic_threshold=fft_threshold,
        min_ac_fraction=min_fraction,
    )

    if not is_saw:
        logger.warning("Sawtooth regime rejected by the ACF/FFT pre-filter.")
        return False, None, None, None, None

    logger.info(f"fft_ratio = {fft_ratio}")

    interval = (0, len(signal))

    return True, interval, estimated_period, period_map, saw_mask
