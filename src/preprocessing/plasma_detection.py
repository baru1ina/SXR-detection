import numpy as np


def detect_plasma_interval(ip_signal, time, threshold_ratio=0.2):
    max_ip = np.max(ip_signal)

    if max_ip <= 0:
        return None

    threshold = threshold_ratio * max_ip

    mask = ip_signal > threshold

    if not np.any(mask):
        return None

    indices = np.where(mask)[0]

    start_idx = indices[0]
    end_idx = indices[-1]

    return time[start_idx], time[end_idx]


def detect_plasma_plateau_mask(
    ip_signal,
    dt,
    level_fraction=0.8,
    hold_s=0.5e-3,
):
    """Return a causal gate that opens after plasma current reaches its plateau.

    The gate deliberately remains open during the later current ramp-down.  Its
    purpose is to remove common-mode startup pickup, not to truncate the end of
    an otherwise valid plasma interval.
    """

    values = np.asarray(ip_signal, dtype=float)
    dt = float(dt)
    if values.ndim != 1 or not len(values):
        raise ValueError("Plasma current must be a non-empty 1D array")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    if not 0 < level_fraction < 1:
        raise ValueError("level_fraction must be between 0 and 1")
    if not np.isfinite(hold_s) or hold_s <= 0:
        raise ValueError("hold_s must be positive and finite")

    finite = values[np.isfinite(values)]
    mask = np.zeros(len(values), dtype=bool)
    normalized = np.full(len(values), np.nan, dtype=float)
    if not len(finite):
        return mask, normalized, None

    plateau_level = float(np.percentile(finite, 95))
    if not np.isfinite(plateau_level) or plateau_level <= 0:
        return mask, normalized, None

    smooth_samples = min(len(values), max(1, int(round(0.25e-3 / dt))))
    kernel = np.ones(smooth_samples, dtype=float) / smooth_samples
    filled = np.where(np.isfinite(values), values, 0.0)
    weights = np.convolve(np.isfinite(values).astype(float), kernel, mode="same")
    smoothed_sum = np.convolve(filled, kernel, mode="same")
    smoothed = np.divide(
        smoothed_sum,
        weights,
        out=np.full(len(values), np.nan, dtype=float),
        where=weights > 0,
    )
    normalized = smoothed / plateau_level

    hold_samples = min(len(values), max(1, int(round(hold_s / dt))))
    above = np.isfinite(normalized) & (normalized >= level_fraction)
    run = np.convolve(above.astype(int), np.ones(hold_samples, dtype=int), mode="valid")
    complete = np.flatnonzero(run >= hold_samples)
    if not len(complete):
        return mask, normalized, None

    start = int(complete[0])
    mask[start:] = True
    return mask, normalized, start
