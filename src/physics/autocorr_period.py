import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, find_peaks

from config.path import PATH_TO_RES_AUTOCORR
from matplotlib.widgets import CheckButtons


def autocorr_fft(x, *, unbiased=True):
    """Return a finite, optionally overlap-corrected autocorrelation."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)

    x = np.where(np.isfinite(x), x, 0.0)
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if unbiased:
        acf = acf / np.maximum(n - np.arange(n), 1)
    return np.asarray(acf, dtype=float)


def autocorr_period(
        signal,
        logger,
        dt,
        min_period=0.001,
        max_period=0.01,
        min_prominence=0.05):
    """Estimate a period only when the ACF contains an interior local peak.

    Linear detrending prevents a monotonic plasma ramp from looking like a
    very short period.  Returning NaN is intentional: an unknown local period
    must not silently turn into a boundary value in the period map.
    """
    x = np.asarray(signal, dtype=float)
    finite = np.isfinite(x)
    if len(x) < 5 or np.sum(finite) < 5:
        return np.nan, 0.0

    if not np.all(finite):
        fill = float(np.median(x[finite]))
        x = np.where(finite, x, fill)
    x = detrend(x, type="linear")
    variance = float(np.var(x))
    if variance <= 1e-12:
        return np.nan, 0.0

    corr = autocorr_fft(x, unbiased=True)
    zero = float(corr[0])
    if not np.isfinite(zero) or zero <= 1e-12:
        return np.nan, 0.0
    corr = corr / zero

    min_lag = max(1, int(np.ceil(min_period / dt)))
    max_lag = min(len(corr) - 1, int(np.floor(max_period / dt)))
    if max_lag - min_lag < 2:
        return np.nan, 0.0

    # Include both limits. find_peaks deliberately cannot select the two
    # boundary samples, which eliminates the common min_period lock.
    segment = corr[min_lag:max_lag + 1]
    peaks, properties = find_peaks(segment, prominence=float(min_prominence))
    if len(peaks) == 0:
        return np.nan, 0.0

    positive = peaks[segment[peaks] > 0]
    if len(positive) == 0:
        return np.nan, 0.0

    peak_idx = int(positive[np.argmax(segment[positive])])
    peak = float(segment[peak_idx])
    prominence_idx = int(np.where(peaks == peak_idx)[0][0])
    prominence = float(properties["prominences"][prominence_idx])
    score = peak * min(1.0, prominence / max(float(min_prominence), 1e-12))
    period = float((min_lag + peak_idx) * dt)
    return period, score


def sliding_autocorr_period(
        signal,
        logger,
        dt,
        window_ms=20,
        step_ms=5,
        min_period=0.001,
        max_period=0.01):

    window = int(window_ms * 1e-3 / dt)
    step = int(step_ms * 1e-3 / dt)

    periods = []
    scores = []
    times = []

    if window < 5 or step < 1 or len(signal) < window:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    for start in range(0, len(signal) - window + 1, step):
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

    return (
        np.asarray(times, dtype=int),
        np.asarray(periods, dtype=float),
        np.asarray(scores, dtype=float),
    )


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
    include_boundary_periods: bool = False,
    support_radius_samples: int = 0,
    max_interp_gap_samples: int | None = None,
    fill_unknown: bool = False,
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

    period_map = np.full(n_samples, np.nan, dtype=float)
    valid_times = np.rint(times[valid]).astype(int)
    valid_periods = periods[valid]
    inside = (valid_times >= 0) & (valid_times < n_samples)
    valid_times = valid_times[inside]
    valid_periods = valid_periods[inside]

    if len(valid_times) == 0:
        if fill_unknown:
            period_map[:] = float(default_period)
        return period_map

    order = np.argsort(valid_times)
    valid_times = valid_times[order]
    valid_periods = valid_periods[order]

    if max_interp_gap_samples is None:
        time_steps = np.diff(valid_times)
        typical_step = int(np.median(time_steps)) if len(time_steps) else 1
        max_interp_gap_samples = max(1, int(round(1.5 * typical_step)))
    else:
        max_interp_gap_samples = max(1, int(max_interp_gap_samples))
    radius = max(0, int(support_radius_samples))

    split_at = np.flatnonzero(np.diff(valid_times) > max_interp_gap_samples) + 1
    groups = np.split(np.arange(len(valid_times)), split_at)
    for group in groups:
        if len(group) == 0:
            continue
        group_times = valid_times[group]
        group_periods = valid_periods[group]
        left = max(0, int(group_times[0]) - radius)
        right = min(n_samples - 1, int(group_times[-1]) + radius)
        query = np.arange(left, right + 1)
        if len(group) == 1:
            values = np.full(len(query), float(group_periods[0]), dtype=float)
        else:
            values = np.interp(query, group_times, group_periods)
        period_map[query] = values

    finite_map = np.isfinite(period_map)
    period_map[finite_map] = np.clip(period_map[finite_map], min_period, max_period)
    if fill_unknown:
        period_map[~finite_map] = float(default_period)
    return period_map


def refine_period_map_from_events(
    n_samples: int,
    event_indices: np.ndarray,
    *,
    dt: float,
    base_period_map: np.ndarray | None = None,
    min_period: float = 1e-3,
    max_period: float = 10e-3,
    min_chain_events: int = 3,
    edge_expansion_periods: float = 0.5,
    max_adjacent_period_ratio: float = 2.25,
) -> np.ndarray:
    """Overwrite an ACF map with periods measured between event candidates.

    Each accepted inter-event interval is kept piecewise constant.  This is
    preferable to smoothing when a short saw packet accelerates abruptly.
    Isolated pairs are ignored; at least ``min_chain_events`` consecutive
    events with physically plausible gaps are required.
    """
    if base_period_map is None:
        refined = np.full(n_samples, np.nan, dtype=float)
    else:
        base = np.asarray(base_period_map, dtype=float)
        refined = base.copy() if len(base) == n_samples else np.full(n_samples, np.nan, dtype=float)

    events = np.asarray(event_indices, dtype=int)
    events = np.unique(events[(events >= 0) & (events < n_samples)])
    if len(events) < max(2, int(min_chain_events)):
        return refined

    gaps = np.diff(events) * float(dt)
    valid_gaps = np.isfinite(gaps) & (gaps >= min_period) & (gaps <= max_period)
    min_chain_gaps = max(1, int(min_chain_events) - 1)

    start = 0
    while start < len(valid_gaps):
        if not valid_gaps[start]:
            start += 1
            continue
        stop = start
        while stop + 1 < len(valid_gaps) and valid_gaps[stop + 1]:
            previous_gap = float(gaps[stop])
            next_gap = float(gaps[stop + 1])
            ratio = max(previous_gap, next_gap) / max(min(previous_gap, next_gap), 1e-12)
            if ratio > float(max_adjacent_period_ratio):
                break
            stop += 1
        if stop - start + 1 >= min_chain_gaps:
            run_gaps = gaps[start:stop + 1]
            first = int(events[start])
            last = int(events[stop + 1])
            left_expand = int(round(edge_expansion_periods * run_gaps[0] / dt))
            right_expand = int(round(edge_expansion_periods * run_gaps[-1] / dt))
            refined[max(0, first - left_expand):first] = float(run_gaps[0])
            for gap_idx in range(start, stop + 1):
                left = int(events[gap_idx])
                right = int(events[gap_idx + 1])
                refined[left:right] = float(gaps[gap_idx])
            refined[last:min(n_samples, last + right_expand + 1)] = float(run_gaps[-1])
        start = stop + 1

    finite = np.isfinite(refined)
    refined[finite] = np.clip(refined[finite], min_period, max_period)
    return refined


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
    expansion_periods: float = 3.0,
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
    if np.sum(robust) == 0:
        print("[build_saw_activity_mask] нет надёжных окон автокорреляции; "
              "активная область может быть восстановлена только по цепочке срывов")
        return mask

    half_window_samples = int(round(0.5 * window_ms * 1e-3 / dt))
    for center, local_period in zip(times[robust], periods[robust]):
        period_samples = int(round(float(local_period) / dt))
        expansion = max(1, half_window_samples + int(round(expansion_periods * period_samples)))
        left = max(0, int(round(center)) - expansion)
        right = min(n_samples, int(round(center)) + expansion + 1)
        mask[left:right] = True
    return mask
