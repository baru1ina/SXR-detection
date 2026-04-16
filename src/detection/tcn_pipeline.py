import numpy as np
import torch

from src.io.loader import SHTLoader
from src.preprocessing.cleaning import remove_mean
from src.preprocessing.normalization import robust_scale
from src.preprocessing.windowing import create_windows
from src.preprocessing.derivative import compute_derivative

from src.ml.models.tcn_predictor import TCNPredictor

from src.anomaly.score import crash_score_derivative
from src.anomaly.smoothing import smooth_score
from src.anomaly.interval_detection import detect_crashes
from src.anomaly.events import group_crashes
from src.visualization.plots import plot_with_crashes
from src.ml.models.model_io import load_model

def detect_on_shot(source_dir,
                   filename,
                   logger,
                   path_to_load="./data/model_data",
                   channel_name="SXR 50 mkm"):

    loader = SHTLoader(source_dir)

    logger.info(f"Loading shot: {filename}")
    shot = loader.load_shot(filename)

    logger.info(f"Total signal length: {len(shot.time)} samples")
    logger.info(f"dt: {shot.dt:.6e} s")

    if channel_name not in shot.channel_names:
        logger.error(f"Channel '{channel_name}' not found in file.")
        return

    if "Ip внутр.(Пр2ВК) (инт.18)" not in shot.channel_names:
        logger.error("Ip channel not found in file.")
        return

    sxr_idx = shot.channel_names.index(channel_name)
    ip_idx = shot.channel_names.index("Ip внутр.(Пр2ВК) (инт.18)")

    sxr_signal = shot.signals[:, sxr_idx:sxr_idx + 1]
    ip_signal = shot.signals[:, ip_idx]

    logger.info("Channels successfully loaded.")

    from src.preprocessing.plasma_detection import detect_plasma_interval

    plasma_interval = detect_plasma_interval(ip_signal, shot.time)

    if plasma_interval is None:
        logger.warning("No plasma interval detected. Abort detection.")
        return

    t_start, t_end = plasma_interval
    logger.info(f"Plasma interval: {t_start:.6f} – {t_end:.6f} s")

    mask = (shot.time >= t_start) & (shot.time <= t_end)

    time_plasma = shot.time[mask]
    sxr_plasma = sxr_signal[mask]

    logger.info(f"Plasma samples: {len(time_plasma)}")

    sxr_plasma = remove_mean(sxr_plasma)
    sxr_plasma = robust_scale(sxr_plasma)

    logger.info(
        f"SXR stats after normalization: "
        f"mean={np.mean(sxr_plasma):.4f}, "
        f"std={np.std(sxr_plasma):.4f}, "
        f"min={np.min(sxr_plasma):.4f}, "
        f"max={np.max(sxr_plasma):.4f}"
    )

    deriv = compute_derivative(sxr_plasma, shot.dt)
    deriv = robust_scale(deriv)

    logger.info(
        f"Derivative stats: "
        f"mean={np.mean(deriv):.4f}, "
        f"std={np.std(deriv):.4f}, "
        f"min={np.min(deriv):.4f}, "
        f"max={np.max(deriv):.4f}"
    )

    window_size = 1000
    logger.info(f"Using fixed window size: {window_size}")

    X, Y = create_windows(deriv, window_size)

    if len(X) == 0:
        logger.warning("No windows created. Abort detection.")
        return

    logger.info(f"Total windows created: {len(X)}")
    logger.info(f"Window tensor shape: {X.shape}")


    logger.info("Loading trained model...")

    model = TCNPredictor(input_dim=1)
    model_path = path_to_load + "/model_data/model_data_cropped.pt"

    try:
        load_model(model, model_path)
    except FileNotFoundError:
        logger.error(f"No trained model found. Abort detection.")
        return
    model.eval()

    logger.info("Model successfully loaded.")
    logger.info("Running model inference...")

    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32))

    preds = preds.numpy()

    logger.info(
        f"Prediction stats: "
        f"mean={np.mean(preds):.4f}, "
        f"std={np.std(preds):.4f}"
    )

    logger.info("Computing anomaly score...")

    score = crash_score_derivative(Y, preds)
    score = smooth_score(score, window=9)

    logger.info(
        f"Score stats: "
        f"mean={np.mean(score):.4f}, "
        f"std={np.std(score):.4f}, "
        f"min={np.min(score):.4f}, "
        f"max={np.max(score):.4f}"
    )

    # threshold = np.mean(score) + ??? * np.std(score)
    threshold = np.percentile(score, 99)
    logger.info(f"Adaptive threshold (mean + 3σ): {threshold:.4f}")

    logger.info(f"Score percentiles: "
                f"90%={np.percentile(score, 90):.3f}, "
                f"95%={np.percentile(score, 95):.3f}, "
                f"99%={np.percentile(score, 99):.3f}")

    crash_indices = detect_crashes(score)
    events = group_crashes(crash_indices)

    logger.info(f"Raw crash indices detected: {len(crash_indices)}")
    logger.info(f"Grouped crash events: {len(events)}")

    crash_times = []

    for event in events:
        idx = window_size + event[0]
        if idx < len(time_plasma):
            crash_times.append(time_plasma[idx])

    logger.info(f"Detected crash times (s): {crash_times}")

    print("Detected crash times:", crash_times)

    plot_with_crashes(shot, crash_times, channel_name, mode="tcn")



