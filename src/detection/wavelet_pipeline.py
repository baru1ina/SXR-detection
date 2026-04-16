import numpy as np

from src.io.loader import SHTLoader

from src.visualization.plots import plot_with_crashes

from src.detection.models.wavelet_detector import WaveletSawtoothDetector
from src.physics.sawtooth_filter import detect_sawtooth_hybrid
from src.physics.autocorr_period import build_period_map
from src.preprocessing.plasma_detection import detect_plasma_interval


def detect_on_shot(source_dir,
                   filename,
                   logger,
                   path_to_load="./data/model_data",
                   channel_name="SXR 50 mkm",
                   wt_threshold=1):

    loader = SHTLoader(source_dir, logger=logger)

    logger.info(f"Разряд: {filename}, толщина фольги: {channel_name}")
    shot = loader.load_shot(filename)

    logger.info(f"Длина сигнала в отсчетах {len(shot.time)}")

    if channel_name not in shot.channel_names:
        logger.error(f"Channel '{channel_name}' not found.")
        return

    if "Ip внутр.(Пр2ВК) (инт.18)" not in shot.channel_names:
        logger.error("Ip (Ip внутр.(Пр2ВК) (инт.18)) channel not found.")
        return

    sxr_idx = shot.channel_names.index(channel_name)
    ip_idx = shot.channel_names.index("Ip внутр.(Пр2ВК) (инт.18)")

    sxr_signal = shot.signals[:, sxr_idx:sxr_idx + 1]
    ip_signal = shot.signals[:, ip_idx]

    logger.info("Channels successfully loaded.")

    plasma_interval = detect_plasma_interval(ip_signal, shot.time)

    if plasma_interval is None:
        logger.warning("No plasma interval detected.")
        return

    t_start, t_end = plasma_interval
    logger.info(f"Plasma interval: {t_start:.6f} – {t_end:.6f} s")

    mask = (shot.time >= t_start) & (shot.time <= t_end)

    time_plasma = shot.time[mask]
    sxr_plasma = sxr_signal[mask]

    logger.info(f"Длина плазменного интервала в отсчетах: {len(time_plasma)}")
    logger.info("Проверка на наличие пилы...")

    signal_1d = sxr_plasma.flatten()

    prehist = sxr_signal[:ip_idx]

    has_saw, interval, estimated_period = detect_sawtooth_hybrid(
        signal_1d,
        prehist,
        dt=shot.dt,
        logger=logger,
        ch=channel_name+"_"+filename
    )

    if not has_saw:
        logger.warning("No sawtooth regime detected: skipping wavelet detection.")
        return

    logger.info(f"Sawtooth detected: {has_saw}")

    if estimated_period is None:
        estimated_period = 3e-3
        logger.warning("No estimated period: using 3 ms")

    logger.info(f"Estimated sawtooth period: {estimated_period:.6e} s")

    logger.info("Wavelet-based reset detection...")

    period_map = build_period_map(signal_1d, logger, shot.dt)

    detector = WaveletSawtoothDetector(
        dt=shot.dt,
        period=estimated_period,
        period_map=period_map,
        crash_time_min=20e-6,
        crash_time_max=100e-6,
        percentile_threshold=98,
        min_period=0.25e-3,
        wavelet_name="gaus1",
        #"mexh"
        wt_threshold=wt_threshold
    )

    peaks, energy, threshold = detector.detect(signal_1d, time_plasma, plot_flag=False)

    # diag = detector.diagnostics()

    # logger.info(f"Scale range used: {diag['scale_range']}")
    # logger.info(
    #     f"Energy stats: mean={diag['energy_mean']:.4f}, "
    #     f"std={diag['energy_std']:.4f}, "
    #     f"max={diag['energy_max']:.4f}"
    # )
    # logger.info(
    #     f"Energy percentiles: 95%={diag['energy_95']:.4f}, "
    #     f"99%={diag['energy_99']:.4f}"
    # )

    logger.info(f"Energy threshold={threshold:.4f}")
    logger.info(f"Raw detected peaks={len(peaks)}")

    margin_time = 0.5e-3
    margin_samples = int(margin_time / shot.dt)

    logger.info(f"Applying edge margin: {margin_time * 1e3:.2f} ms "
                f"({margin_samples} samples)")

    valid_mask = np.zeros_like(signal_1d, dtype=bool)

    if len(signal_1d) > 2 * margin_samples:
        valid_mask[margin_samples:-margin_samples] = True
    else:
        logger.warning("Signal too short for margin filtering.")

    filtered_peaks = peaks[valid_mask[peaks]]

    logger.info(f"Peaks after edge filtering: {len(filtered_peaks)}")

    crash_times = time_plasma[filtered_peaks]

    logger.info(f"Final detected reset count: {len(crash_times)}")
    logger.info(f"Detected reset times: {crash_times}")

    plot_with_crashes(shot, crash_times, channel_name, mode="wavelet")

