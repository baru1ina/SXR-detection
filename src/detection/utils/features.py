from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import medfilt, savgol_filter
from scipy.stats import kurtosis, skew


_EPS = 1e-12


def make_odd(value: int, minimum: int = 3) -> int:
    value = max(int(value), minimum)
    return value if value % 2 else value + 1


def mad_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return float(1.4826 * mad + _EPS)


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - np.nanmedian(x)) / mad_std(x)


def safe_standardize_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=0)
    scale = np.array([mad_std(x[:, i]) for i in range(x.shape[1])])
    return (x - med) / (scale + _EPS)


def smooth_signal(x: np.ndarray, median_window: int = 31, gaussian_sigma: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return x.copy()
    k = make_odd(min(median_window, max(3, len(x) // 10)))
    y = medfilt(x, kernel_size=k)
    if gaussian_sigma and gaussian_sigma > 0:
        y = gaussian_filter1d(y, sigma=gaussian_sigma, mode="nearest")
    return y


def detrend_savgol(x: np.ndarray, dt: float, window_s: Optional[float] = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 11:
        return x - np.mean(x)
    if window_s is None:
        window_s = min(5e-3, max(1e-3, 0.05 * len(x) * dt))
    w = make_odd(int(window_s / dt), minimum=11)
    w = min(w, make_odd(len(x) - 2 if len(x) % 2 == 0 else len(x), minimum=11))
    if w >= len(x):
        w = make_odd(len(x) - 2, minimum=5)
    if w < 5:
        return x - np.mean(x)
    baseline = savgol_filter(x, window_length=w, polyorder=min(3, w - 2), mode="interp")
    return x - baseline


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    return uniform_filter1d(np.asarray(x, dtype=float), size=window, mode="nearest")


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = rolling_mean(x, window)
    m2 = rolling_mean(x * x, window)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def local_energy(x: np.ndarray, window: int) -> np.ndarray:
    x = robust_zscore(x)
    return rolling_mean(x * x, max(int(window), 1))


def build_cpd_feature_matrix(
    signal: np.ndarray,
    dt: float,
    period: Optional[float] = None,
    include_raw: bool = True,
) -> np.ndarray:
    x = smooth_signal(signal)
    x_centered = robust_zscore(x)
    x_detrended = robust_zscore(detrend_savgol(x, dt))

    dx = np.gradient(x_detrended, dt)
    dx_z = robust_zscore(dx)
    neg_dx = np.maximum(-dx_z, 0.0)
    abs_dx = np.abs(dx_z)

    if period is None or not np.isfinite(period) or period <= 0:
        energy_window = max(5, int(0.25e-3 / dt))
    else:
        energy_window = max(5, int(0.08 * period / dt))

    e_dx = robust_zscore(local_energy(dx_z, energy_window))
    fast_minus_slow = robust_zscore(x - smooth_signal(x, median_window=max(51, 4 * energy_window + 1), gaussian_sigma=2.0))

    cols = []
    if include_raw:
        cols.append(x_centered)
    cols.extend([x_detrended, dx_z, neg_dx, abs_dx, e_dx, fast_minus_slow])
    features = np.column_stack(cols)
    return safe_standardize_2d(features)


def _window(signal: np.ndarray, center: int, radius: int) -> np.ndarray:
    left = max(0, center - radius)
    right = min(len(signal), center + radius + 1)
    return np.asarray(signal[left:right], dtype=float)


def score_crash_candidate(
    signal: np.ndarray,
    index: int,
    dt: float,
    period: Optional[float] = None,
    min_window_s: float = 50e-6,
    max_window_s: float = 700e-6,
    left_factor: float = 0.18,
    right_factor: float = 0.12,
) -> Tuple[float, str, dict]:
    x = np.asarray(signal, dtype=float)
    n = len(x)
    i = int(index)
    if i <= 1 or i >= n - 2:
        return 0.0, "edge", {}

    if period is None or not np.isfinite(period) or period <= 0:
        left_s = right_s = 250e-6
    else:
        left_s = np.clip(left_factor * period, min_window_s, max_window_s)
        right_s = np.clip(right_factor * period, min_window_s, max_window_s)

    left_n = max(3, int(left_s / dt))
    right_n = max(3, int(right_s / dt))

    pre = x[max(0, i - left_n):i]
    post = x[i:min(n, i + right_n)]
    around = x[max(0, i - left_n):min(n, i + right_n)]
    if len(pre) < 3 or len(post) < 3 or len(around) < 5:
        return 0.0, "edge", {}

    noise = mad_std(np.diff(around))
    amp_noise = mad_std(around)

    normal_drop = np.percentile(pre, 90) - np.percentile(post, 10)
    inverted_rise = np.percentile(post, 90) - np.percentile(pre, 10)
    amp_change = max(normal_drop, inverted_rise)

    deriv = np.gradient(around, dt)
    max_neg = -float(np.min(deriv))
    max_pos = float(np.max(deriv))
    deriv_scale = mad_std(deriv)
    slope_score = max(max_neg, max_pos) / (deriv_scale + _EPS)

    # local_amp is a robust peak-to-peak estimate in the local neighbourhood.
    # It is intentionally different from amp_noise: amp_noise measures scatter,
    # while local_amp measures the full local oscillation amplitude.  The ratio
    # jump_to_amp answers: "how large is the candidate jump compared with the
    # local sawtooth amplitude?".
    local_amp = float(np.percentile(around, 95) - np.percentile(around, 5))
    jump_to_amp = float(max(0.0, amp_change) / (local_amp + _EPS))

    amp_score = amp_change / (amp_noise + _EPS)
    score = float(max(0.0, amp_score) + 0.15 * slope_score)

    if normal_drop > inverted_rise * 1.15:
        direction = "normal"
    elif inverted_rise > normal_drop * 1.15:
        direction = "inverted"
    else:
        direction = "bidirectional"

    meta = {
        "normal_drop": float(normal_drop),
        "inverted_rise": float(inverted_rise),
        "amp_score": float(amp_score),
        "slope_score": float(slope_score),
        "local_amp": float(local_amp),
        "jump_to_amp": float(jump_to_amp),
        "left_samples": int(left_n),
        "right_samples": int(right_n),
    }
    return score, direction, meta


def suppress_by_period(
    indices: Sequence[int],
    scores: Sequence[float],
    min_distance: int,
) -> np.ndarray:
    if len(indices) == 0:
        return np.array([], dtype=int)
    idx = np.asarray(indices, dtype=int)
    sc = np.asarray(scores, dtype=float)
    order = np.argsort(sc)[::-1]
    kept: list[int] = []
    min_distance = max(int(min_distance), 1)
    for j in order:
        candidate = idx[j]
        if all(abs(candidate - k) >= min_distance for k in kept):
            kept.append(int(candidate))
    kept.sort()
    return np.array(kept, dtype=int)


def extract_candidate_features(
    signal: np.ndarray,
    index: int,
    dt: float,
    period: Optional[float] = None,
) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    score, direction, meta = score_crash_candidate(x, index, dt, period)

    if period is None or not np.isfinite(period) or period <= 0:
        radius = max(10, int(0.5e-3 / dt))
    else:
        radius = max(10, int(np.clip(0.25 * period, 0.2e-3, 2e-3) / dt))
    seg = _window(x, index, radius)
    dseg = np.gradient(seg, dt) if len(seg) >= 3 else np.zeros_like(seg)

    pre = x[max(0, index - radius):index]
    post = x[index:min(len(x), index + radius)]
    if len(pre) == 0:
        pre = seg
    if len(post) == 0:
        post = seg

    feats = [
        score,
        meta.get("amp_score", 0.0),
        meta.get("slope_score", 0.0),
        meta.get("normal_drop", 0.0),
        meta.get("inverted_rise", 0.0),
        float(np.mean(seg)),
        float(np.std(seg)),
        float(mad_std(seg)),
        float(np.max(seg) - np.min(seg)),
        float(np.min(dseg)),
        float(np.max(dseg)),
        float(np.std(dseg)),
        float(kurtosis(seg, fisher=False, bias=False)) if len(seg) > 4 else 0.0,
        float(skew(seg, bias=False)) if len(seg) > 4 else 0.0,
        float(np.mean(post) - np.mean(pre)),
        float(np.std(post) / (np.std(pre) + _EPS)),
    ]
    return np.nan_to_num(np.asarray(feats, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
