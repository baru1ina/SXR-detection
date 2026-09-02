from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from config.path import IP_CHANNEL, SXR_CHANNELS
from src.detection.utils.multichannel import MultiChannelVotingDetector
from src.io.loader import SHTLoader
from src.physics.sawtooth_filter import detect_sawtooth_hybrid
from src.preprocessing.cleaning import remove_mean
from src.preprocessing.normalization import robust_scale
from src.preprocessing.plasma_detection import detect_plasma_interval
from src.visualization.plots import plot_with_crashes


@dataclass
class PreparedShot:
    shot: object
    filename: str
    channel_name: str
    time_plasma: np.ndarray
    reference_signal: np.ndarray
    channel_signals: Dict[str, np.ndarray]
    dt: float
    estimated_period: float
    mask: np.ndarray
    period_map: Optional[np.ndarray] = None
    saw_mask: Optional[np.ndarray] = None


def preprocess_sxr_signal(signal: np.ndarray) -> np.ndarray:
    return robust_scale(remove_mean(np.asarray(signal, dtype=float)))


def safe_prehistory(time: np.ndarray, sxr_signal: np.ndarray, t_start: float) -> np.ndarray:
    pre_mask = np.asarray(time) < float(t_start)
    prehist = np.asarray(sxr_signal)[pre_mask]
    if len(prehist) >= 10:
        return prehist

    n = max(10, min(len(sxr_signal) // 20, 2000))
    return np.asarray(sxr_signal)[:n]


def prepare_shot_for_detection(
    source_dir: str,
    filename: str,
    logger,
    channel_name: str = "SXR 50 mkm",
    channels: Optional[Iterable[str]] = None,
) -> Optional[PreparedShot]:
    requested_channels = list(channels or SXR_CHANNELS)
    load_channels = sorted(set(requested_channels + [channel_name, IP_CHANNEL]))

    loader = SHTLoader(source_dir, logger=logger)
    shot = loader.load_shot(filename, channels=load_channels)
    logger.info(f"Разряд: {filename}, канал: {channel_name}")
    logger.info(f"Загруженные каналы: {shot.channel_names}")

    if IP_CHANNEL not in shot.channel_names:
        logger.error(f"Ip channel '{IP_CHANNEL}' not found.")
        logger.newline()
        return None
    if channel_name not in shot.channel_names:
        logger.error(f"Channel '{channel_name}' not found.")
        logger.newline()
        return None

    ip_signal = shot.signals[:, shot.channel_names.index(IP_CHANNEL)]
    plasma_interval = detect_plasma_interval(ip_signal, shot.time)
    if plasma_interval is None:
        logger.warning("No plasma interval detected.")
        return None

    t_start, t_end = plasma_interval
    logger.info(f"Plasma interval: {t_start:.6f} – {t_end:.6f} s")

    mask = (shot.time >= t_start) & (shot.time <= t_end)
    time_plasma = shot.time[mask]
    dt = float(shot.dt)

    ref_raw = shot.signals[:, shot.channel_names.index(channel_name)]
    reference_signal = preprocess_sxr_signal(ref_raw[mask])
    prehist = safe_prehistory(shot.time, ref_raw, t_start)

    saw_result = detect_sawtooth_hybrid(
        reference_signal,
        prehist,
        dt=dt,
        logger=logger,
        ch=f"{channel_name}_{filename}",
        signal_for_snr=ref_raw[mask],
    )
    if len(saw_result) == 5:
        has_saw, _, estimated_period, period_map, saw_mask = saw_result
    else:
        has_saw, _, estimated_period, period_map = saw_result
        saw_mask = None
    if not has_saw:
        logger.warning(
            "Shot/channel rejected by the preliminary sawtooth check; "
            "the downstream detector and event-based period refinement will not run."
        )
        logger.newline()
        return None

    # if has_saw:
    #     print(f"[TIME_PLASMA]: {time_plasma}")
    #     print(f"[SAW_MASK]: {saw_mask}")

    if estimated_period is None or not np.isfinite(estimated_period) or estimated_period <= 0:
        estimated_period = 3e-3
        logger.warning("No estimated sawtooth period: using 3 ms.")

    # estimated_period = 3e-3

    logger.info(f"Estimated sawtooth period: {estimated_period:.6e} s")

    channel_signals: Dict[str, np.ndarray] = {}
    for ch in requested_channels:
        if ch not in shot.channel_names:
            # logger.warning(f"Skipping missing SXR channel: {ch}")
            continue
        raw = shot.signals[:, shot.channel_names.index(ch)]
        channel_signals[ch] = preprocess_sxr_signal(raw[mask])

    if channel_name not in channel_signals:
        channel_signals[channel_name] = reference_signal

    return PreparedShot(
        shot=shot,
        filename=filename,
        channel_name=channel_name,
        time_plasma=time_plasma,
        reference_signal=reference_signal,
        channel_signals=channel_signals,
        dt=dt,
        estimated_period=float(estimated_period),
        mask=mask,
        period_map=period_map,
        saw_mask=saw_mask,
    )


def run_single_channel_detector(
    prepared: PreparedShot,
    detector,
    logger,
    downsample: int = 1,
    plot: bool = True,
    mode: str = "detector",
    debug: bool = False,
) -> np.ndarray:
    ds = downsample
    signal_ds = prepared.reference_signal[::ds]
    dt_ds = prepared.dt * ds

    period_map_ds = None
    if prepared.period_map is not None:
        period_map_ds = prepared.period_map[::ds]

    active_mask_ds = None
    if prepared.saw_mask is not None:
        active_mask_ds = prepared.saw_mask[::ds]

    indices_ds = detector.detect(
        signal_ds,
        period=prepared.estimated_period,
        channel=prepared.channel_name,
        debug=debug,
        period_map=period_map_ds,
        active_mask=active_mask_ds,
    )
    indices = (np.asarray(indices_ds, dtype=int) * ds).astype(int)
    indices = indices[(indices >= 0) & (indices < len(prepared.time_plasma))]
    effective_saw_mask = prepared.saw_mask
    detector_period_map = getattr(detector, "last_period_map", None)
    if detector_period_map is not None and len(detector_period_map) == len(signal_ds):
        event_support = np.repeat(np.isfinite(detector_period_map), ds)[:len(prepared.time_plasma)]
        if effective_saw_mask is None:
            effective_saw_mask = event_support
        else:
            effective_saw_mask = np.asarray(effective_saw_mask, dtype=bool) | event_support
    if effective_saw_mask is not None and len(indices) > 0:
        indices = indices[effective_saw_mask[indices]]
    crash_times = prepared.time_plasma[indices]

    logger.info(f"Single-channel detections: {len(crash_times)}")
    logger.info(f"Detected reset times: {crash_times}")
    logger.newline()

    if plot and len(crash_times) > 0:
        plot_with_crashes(prepared.shot, crash_times, channel_name=prepared.channel_name, mode=mode)

    if debug and hasattr(detector, "last_energy") and detector.last_energy is not None:
        try:
            from src.visualization.plots import plot_signal_energy_period_map

            ds = max(1, int(downsample))
            time_ds = prepared.time_plasma[::ds]
            period_map_acf_ds = prepared.period_map[::ds] if prepared.period_map is not None else None
            period_map_ds = period_map_acf_ds
            detector_period_map = getattr(detector, "last_period_map", None)
            if detector_period_map is not None and len(detector_period_map) == len(signal_ds):
                period_map_ds = detector_period_map
            active_mask_ds = effective_saw_mask[::ds] if effective_saw_mask is not None else None
            plot_signal_energy_period_map(
                time=time_ds,
                signal=signal_ds,
                energy=detector.last_energy,
                period_map=period_map_ds,
                period_map_acf=period_map_acf_ds,
                crash_times=crash_times,
                active_mask=active_mask_ds,
                channel_name=prepared.channel_name,
                filename=prepared.filename,
                mode=f"{mode}_diagnostics",
            )
        except Exception as exc:
            logger.warning(f"Could not plot wavelet period-map diagnostics: {exc}")

    return crash_times


def run_multichannel_detector(
    prepared: PreparedShot,
    detector_factory,
    logger,
    downsample: int = 1,
    coincidence_window: float = 0.3e-3,
    min_channels: int = 2,
    plot: bool = True,
    mode: str = "detector_multichannel",
    debug: bool = False,
) -> np.ndarray:
    ds = max(1, int(downsample))
    dt_ds = prepared.dt * ds

    period_map_ds = None
    if prepared.period_map is not None:
        period_map_ds = prepared.period_map[::ds]

    active_mask_ds = None
    if prepared.saw_mask is not None:
        active_mask_ds = prepared.saw_mask[::ds]

    signals_ds = {ch: sig[::ds] for ch, sig in prepared.channel_signals.items()}
    voter = MultiChannelVotingDetector(
        detector_factory=detector_factory,
        dt=dt_ds,
        coincidence_window=coincidence_window,
        min_channels=min_channels,
    )
    indices_ds = voter.detect(
        signals_ds,
        period=prepared.estimated_period,
        debug=debug,
        period_map=period_map_ds,
        active_mask=active_mask_ds,
    )
    indices = (np.asarray(indices_ds, dtype=int) * ds).astype(int)
    indices = indices[(indices >= 0) & (indices < len(prepared.time_plasma))]
    # Each per-channel detector has already applied the ACF mask combined
    # with its own event-refined period support before channel voting.
    crash_times = prepared.time_plasma[indices]

    logger.info(f"Multichannel detections: {len(crash_times)}")
    logger.info(f"Per-channel candidates: { {ch: len(c) for ch, c in voter.channel_candidates.items()} }")
    logger.info(f"Detected reset times: {crash_times}")
    logger.newline()

    if plot and len(crash_times) > 0:
        plot_with_crashes(prepared.shot, crash_times, channel_name=prepared.channel_name, mode=mode)
    return crash_times

