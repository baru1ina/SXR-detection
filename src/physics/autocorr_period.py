import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from config.path import PATH_TO_RES_AUTOCORR
from matplotlib.widgets import CheckButtons


def autocorr_fft(x):
    n = len(x)
    f = np.fft.rfft(x, n=2*n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf


def autocorr_period(signal, logger, dt, min_period=0.001, max_period=0.01):

    # logger.info(f"Дебаг autocorr_period")

    # n = len(signal)
    # corr = np.correlate(signal, signal, mode="full")
    # corr = corr[len(corr)//2:]

    corr = autocorr_fft(signal)

    variance = np.var(signal) * len(signal)
    corr = corr / variance if variance != 0 else corr

    min_lag = int(min_period / dt)
    max_lag = int(max_period / dt)

    segment = corr[min_lag:max_lag]

    if len(segment) == 0:
        return None, 0

    peak_idx = np.argmax(segment)
    peak = segment[peak_idx]

    zero = corr[0]

    score = peak / zero if zero != 0 else 0

    period = (min_lag + peak_idx) * dt

    return period, score


def sliding_autocorr_period(
        signal,
        logger,
        dt,
        window_ms=20,
        step_ms=5,
        min_period=0.001,
        max_period=0.01):

    # logger.info(f"Дебаг sliding_autocorr_period")

    signal = signal - np.mean(signal)

    window = int(window_ms * 1e-3 / dt)
    step = int(step_ms * 1e-3 / dt)

    periods = []
    scores = []
    times = []

    for start in range(0, len(signal) - window, step):

        seg = signal[start:start+window]

        period, score = autocorr_period(
            seg,
            logger,
            dt,
            min_period,
            max_period
        )
        periods.append(period)
        scores.append(score)

        center = start + window // 2
        times.append(center)

    logger.info(f"Карта периодов в пиле: {periods}")
    logger.info(f"Значения score: {scores}")

    return np.array(times), np.array(periods), np.array(scores)


def build_period_map(signal, logger, dt):

    times, periods, scores = sliding_autocorr_period(signal, logger, dt)

    valid = ~np.isnan(periods)

    if np.sum(valid) < 3:
        return None

    f = interp1d(
        times,
        periods,
        bounds_error=False,
        fill_value="extrapolate"
    )

    idx = np.arange(len(signal))

    return f(idx)