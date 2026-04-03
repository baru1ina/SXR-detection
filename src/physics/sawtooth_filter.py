import numpy as np
import matplotlib.pyplot as plt

from config.path import PATH_TO_RES_AUTOCORR
from matplotlib.widgets import CheckButtons

from .autocorr_period import autocorr_score, sliding_autocorr_period, autocorr_period
from .fft import fft_score, fft_score_improved, fft_score_debug, fft_score_with_drift
from scipy.stats import skew

from src.anomaly.smoothing import smooth_signal_median

def is_sawtooth(interval, shot, drop_threshold=2.0):
    start, end = interval
    segment = shot.signals[start:end]
    diff = np.diff(segment, axis=0)
    max_drop = np.min(diff)
    return max_drop < -drop_threshold


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


def detect_sawtooth_hybrid_v0(signal, dt, min_cycles=3):

    n = len(signal)

    if n < 1000:
        return False, None, None, 0.0

    signal = signal - np.mean(signal)

    ac_score, period = autocorr_score(signal, dt)

    if period is None:
        return False, None, None, 0.0

    ac_norm = ac_score / (np.std(signal) + 1e-8)

    fft_ratio = fft_score(signal, dt)

    fft_norm = fft_ratio / (np.std(signal) + 1e-8)

    duration_sec = n * dt

    if duration_sec < min_cycles * period:
        return False, None, None, 0.0


    confidence = 0.5 * ac_norm + 0.5 * fft_norm

    threshold = np.median([ac_norm, fft_norm])

    has_saw = confidence > threshold

    return has_saw, (0, n), period, confidence


def detect_sawtooth_hybrid_v2(signal, dt,
                           ac_threshold=0.3,
                           fft_threshold=3,
                           skew_threshold=-0.5,
                           min_duration_ms=20):

    n = len(signal)

    if n < 1000:
        return False, None, None

    signal = signal - np.mean(signal)

    # --- AC ---
    ac_score, period = autocorr_score(signal, dt)

    # --- FFT ---
    fft_ratio = fft_score(signal, dt)

    # --- асимметрия ---
    deriv = np.gradient(signal, dt)
    sk = skew(deriv)

    duration = n * dt * 1000

    if duration < min_duration_ms:
        return False, None, None

    if (
        ac_score > ac_threshold
        and fft_ratio > fft_threshold
        and sk < skew_threshold
    ):
        return True, (0, n), period

    return False, None, None


def detect_sawtooth_hybrid_v3(signal, dt, logger, ch,
                           ac_threshold=0.3,
                           fft_threshold=3,
                           min_duration_ms=20):
    n = len(signal)

    if n < 1000:
        return False, None, None

    signal = signal - np.mean(signal)

    # --- автокорреляция ---
    ac_score, period = autocorr_score(signal, dt, logger=logger, ch=ch)

    logger.info(f"ac_score: {ac_score}")
    logger.info(f"ac_threshold: {ac_threshold}")
    logger.info(f"period: {period}")

    # --- FFT ---
    fft_ratio = fft_score(signal, dt)
    logger.info(f"fft_ratio: {fft_ratio}")

    # --- длительность режима ---
    duration = n * dt * 1000
    logger.info(f"duration: {duration}")

    if duration < min_duration_ms:
        return False, None, None

    if ac_score > ac_threshold and fft_ratio > fft_threshold:
        return True, (0, n), period

    return False, None, None


from scipy.signal import butter, filtfilt

def highpass_filter(signal, dt, cutoff_freq=50):
    """
    Удаляет низкочастотный тренд
    """
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
        fft_threshold=0.05,
        min_fraction=0.5,
        min_snr=0.9):

    plt.figure(figsize=(12, 8))
    plt.plot(np.linspace(0, len(signal) * dt * 1000, len(signal)), signal)
    signal = highpass_filter(signal, dt, cutoff_freq=100)
    plt.plot(np.linspace(0, len(signal) * dt * 1000, len(signal)), signal)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    noise_estimate = np.std(prehist)
    signal_amplitude = np.std(signal)
    snr = signal_amplitude / (noise_estimate + 1e-8)
    logger.info(f"Signal-to-noise ratio = {snr:.2f}")

    if snr < min_snr:
        logger.info(f"SNR too low: {snr:.2f} < {min_snr}")
        return False, None, None

    times, periods, scores = sliding_autocorr_period(
        signal,
        logger,
        dt
    )

    valid = scores > ac_threshold

    fraction = np.sum(valid) / len(scores)

    logger.info(f"sawtooth windows fraction = {fraction:.3f}")

    estimated_period = np.median(periods[valid])

    logger.info(f"estimated_period = {estimated_period}")

    fft_ratio, ratio_2_1, freq_evolution = fft_score_with_drift(
        signal, logger, dt, times, periods, scores
    )

    logger.info(f"fft_ratio = {fft_ratio}")

    if fft_ratio < fft_threshold:
        logger.info(f"fft_ratio is invalid.")
        return False, None, None

    interval = (0, len(signal))

    return True, interval, estimated_period