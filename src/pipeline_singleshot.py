import torch
from torch.utils.data import DataLoader

from src.io.loader import SHTLoader
from src.preprocessing.cleaning import remove_mean
from src.preprocessing.normalization import robust_scale
from src.preprocessing.windowing import create_windows, estimate_window_size
from src.preprocessing.derivative import compute_derivative

from src.ml.dataset import WindowDataset
from src.ml.models.tcn_predictor import TCNPredictor
from src.ml.trainer import train

from src.anomaly.score import crash_score_derivative
from src.anomaly.smoothing import smooth_score
from src.anomaly.interval_detection import detect_crashes
from src.anomaly.events import group_crashes
from src.visualization.plots import plot_with_crashes

from src.logger import setup_logger

def run_pipeline(source_dir, filename):

    logger = setup_logger()

    loader = SHTLoader(source_dir)
    logger.info("Loading shots...")

    shot = loader.load_shot(filename)

    signals = remove_mean(shot.signals)
    signals = robust_scale(signals)

    logger.info(f"Signal length: {len(signals)}")

    deriv = compute_derivative(signals, shot.dt)
    deriv = robust_scale(deriv)

    window_size = estimate_window_size(shot, approx_period_ms=2.0)

    logger.info(f"Window size estimated: {window_size}")
    logger.info("Building dataset...")

    X, Y = create_windows(deriv, window_size)

    logger.info(f"Total windows: {len(X)}")

    dataset = WindowDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = TCNPredictor(input_dim=signals.shape[1])

    logger.info("Starting training...")
    train(model, dataloader, logger, epochs=15)

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32))

    preds = preds.numpy()

    score = crash_score_derivative(Y, preds)

    score = smooth_score(score, window=9)

    crash_indices = detect_crashes(score, k=4)

    events = group_crashes(crash_indices, min_gap=window_size // 3)

    crash_times = [
        shot.time[window_size + event[0]]
        for event in events
    ]

    print("Detected crash times:", crash_times)

    plot_with_crashes(shot, crash_times)

    return crash_times