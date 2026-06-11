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


def robust_period_from_map(
    periods: np.ndarray,
    scores: np.ndarray,
    *,
    ac_threshold: float = 0.3,
    min_period: float = 1e-3,
    max_period: float = 10e-3,
    boundary_eps: float = 0.10,
    default: float = 4e-3,
) -> float:
    periods = np.asarray(periods, dtype=float)
    scores = np.asarray(scores, dtype=float)

    finite = np.isfinite(periods) & np.isfinite(scores)

    valid = (
        finite
        & (scores > ac_threshold)
        & (periods > min_period * (1.0 + boundary_eps))
        & (periods < max_period * (1.0 - boundary_eps))
    )

    if np.sum(valid) >= 3:
        p = periods[valid]
        w = scores[valid]
        order = np.argsort(p)
        p_sorted = p[order]
        w_sorted = w[order]
        cumulative = np.cumsum(w_sorted)
        cutoff = 0.5 * cumulative[-1]
        return float(p_sorted[np.searchsorted(cumulative, cutoff)])

    valid = finite & (scores > ac_threshold)
    if np.sum(valid) >= 3:
        return float(np.median(periods[valid]))

    return float(default)


def build_period_map(
    n_samples: int,
    times: np.ndarray,
    periods: np.ndarray,
    scores: np.ndarray,
    *,
    ac_threshold: float = 0.3,
    min_period: float = 1e-3,
    max_period: float = 10e-3,
    boundary_eps: float = 0.10,
    default_period: float = 4e-3,
    include_boundary_periods: bool = True,
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    periods = np.asarray(periods, dtype=float)
    scores = np.asarray(scores, dtype=float)

    finite = np.isfinite(times) & np.isfinite(periods) & np.isfinite(scores)
    valid = finite & (scores > ac_threshold)

    if not include_boundary_periods:
        valid = (
            valid
            & (periods > min_period * (1.0 + boundary_eps))
            & (periods < max_period * (1.0 - boundary_eps))
        )

    idx = np.arange(n_samples)

    if np.sum(valid) < 2:
        return np.full(n_samples, float(default_period), dtype=float)

    f = interp1d(
        times[valid],
        periods[valid],
        bounds_error=False,
        fill_value=(periods[valid][0], periods[valid][-1]),
    )

    period_map = f(idx)
    period_map = np.where(np.isfinite(period_map), period_map, default_period)
    period_map = np.clip(period_map, min_period, max_period)
    return period_map.astype(float)


def build_saw_activity_mask(
    n_samples: int,
    times: np.ndarray,
    periods: np.ndarray,
    scores: np.ndarray,
    *,
    dt: float,
    global_period: float,
    ac_threshold: float = 0.3,
    min_period: float = 1e-3,
    max_period: float = 10e-3,
    boundary_eps: float = 0.10,
    window_ms: float = 20,
    expansion_periods: float = 1.0,
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    periods = np.asarray(periods, dtype=float)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(times) & np.isfinite(periods) & np.isfinite(scores)

    robust = (
        finite
        & (scores > ac_threshold)
        & (periods > min_period * (1.0 + boundary_eps))
        & (periods < max_period * (1.0 - boundary_eps))
    )

    mask = np.zeros(n_samples, dtype=bool)
    if np.sum(robust) < 2:
        print("[build_saw_activity_mask] сработало ограничение на количество надёжных окон автокорреляции, "
              "берем весь диапазон плазменного шнура")
        mask[:] = True
        return mask

    half_window_samples = int(round(0.5 * window_ms * 1e-3 / dt))
    period_samples = int(round(float(global_period) / dt))
    expansion = max(1, half_window_samples + int(round(expansion_periods * period_samples)))

    left = max(0, int(np.min(times[robust])) - expansion)
    right = min(n_samples, int(np.max(times[robust])) + expansion + 1)
    mask[left:right] = True
    return mask
