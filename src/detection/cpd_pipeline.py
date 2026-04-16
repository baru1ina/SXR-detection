import numpy as np

from src.io.loader import SHTLoader
from src.preprocessing.cleaning import remove_mean
from src.preprocessing.normalization import robust_scale

from src.visualization.plots import plot_with_crashes

from src.detection.models.cpd_detector import CPDDetector
from src.physics.sawtooth_filter import detect_sawtooth_hybrid
from src.preprocessing.plasma_detection import detect_plasma_interval
from src.physics.autocorr_period import build_period_map


def detect_on_shot(
    source_dir,
    filename,
    logger,
    path_to_load="./data/model_data",
    channel_name="SXR 50 mkm",
    penalty=3
):

    loader = SHTLoader(source_dir, logger=logger)

    logger.info(f"Разряд: {filename}, канал: {channel_name}")
    shot = loader.load_shot(filename)

    logger.info(f"Длина сигнала: {len(shot.time)}")

    if channel_name not in shot.channel_names:
        logger.error(f"Channel '{channel_name}' not found.")
        return

    if "Ip внутр.(Пр2ВК) (инт.18)" not in shot.channel_names:
        logger.error("Ip channel not found.")
        return

    sxr_idx = shot.channel_names.index(channel_name)
    ip_idx = shot.channel_names.index("Ip внутр.(Пр2ВК) (инт.18)")

    sxr_signal = shot.signals[:, sxr_idx]
    ip_signal = shot.signals[:, ip_idx]

    logger.info("Channels successfully loaded.")

    plasma_interval = detect_plasma_interval(ip_signal, shot.time)

    if plasma_interval is None:
        logger.warning("No plasma interval detected.")
        return

    t_start, t_end = plasma_interval
    # t_start, t_end = 0.18, 0.21
    logger.info(f"Plasma interval: {t_start:.6f} – {t_end:.6f} s")

    mask = (shot.time >= t_start) & (shot.time <= t_end)

    time_plasma = shot.time[mask]
    sxr_plasma = sxr_signal[mask]

    logger.info(f"Длина плазменного интервала: {len(time_plasma)}")

    signal_1d = sxr_plasma.copy()
    signal_1d = remove_mean(signal_1d)
    signal_1d = robust_scale(signal_1d)

    logger.info("Проверка на наличие пилы...")

    prehist = sxr_signal[:ip_idx]

    has_saw, interval, estimated_period = detect_sawtooth_hybrid(
        signal_1d,
        prehist,
        dt=shot.dt,
        logger=logger,
        ch=channel_name + "_" + filename
    )

    if not has_saw:
        logger.warning("No sawtooth regime detected: skipping CPD.")
        return

    logger.info(f"Sawtooth detected: {has_saw}")

    if estimated_period is None:
        estimated_period = 3e-3
        logger.warning("No estimated period: using 3 ms")

    logger.info(f"Estimated period: {estimated_period:.6e} s")

    logger.info("CPD-based reset detection...")
    downsample = 10
    signal_ds = signal_1d[::downsample]
    dt_ds = shot.dt * downsample

    detector = CPDDetector(
        dt=dt_ds,
        # penalty=penalty,
        model="rbf",
        # model="l2",
        min_distance_factor=0.3,
        drop_window_factor=0.3,
        drop_threshold_factor=0.5
    )

    period_map = build_period_map(signal_1d, logger, shot.dt)

    reset_idx = detector.detect_old(
        signal_ds,
        period=estimated_period,
        # period_map=period_map,
        debug=False
    )

    reset_times = (reset_idx * downsample).astype(int)

    logger.info(f"Raw CPD resets: {len(reset_times)}")

    if len(reset_times) == 0:
        logger.warning("No resets detected by CPD.")
        return

    margin_time = 0.5e-3
    margin_samples = int(margin_time / shot.dt)

    logger.info(f"Applying edge margin: {margin_time * 1e3:.2f} ms "
                f"({margin_samples} samples)")

    valid_mask = (reset_times >= margin_samples) & (reset_times <= len(signal_1d) - margin_samples)
    reset_times_filtered = reset_times[valid_mask]

    reset_times_plasma = time_plasma[reset_times]

    logger.info(f"Resets after edge filtering: {len(reset_times_filtered)}")
    logger.info(f"Detected reset times: {reset_times_filtered}")

    plot_with_crashes(shot, reset_times_plasma, channel_name, mode="cpd")