import numpy as np
import os

from torch.ao.quantization.pt2e.duplicate_dq_pass import logger
from torch.utils.data import DataLoader

from src.io.loader import SHTLoader
from src.ml.builder import build_dataset
from src.ml.dataset import WindowDataset
from src.ml.models.tcn_predictor import TCNPredictor
from src.ml.models.model_io import save_model
from src.ml.trainer import train
from src.preprocessing.cleaning import remove_mean
from src.preprocessing.normalization import robust_scale
from src.preprocessing.windowing import create_windows, estimate_window_size
from src.preprocessing.derivative import compute_derivative
from src.ml.split import split_shots
from src.preprocessing.plasma_detection import detect_plasma_interval
from src.physics.sawtooth_filter import detect_sawtooth_hybrid, detect_sawtooth_by_crashes

from config.path import IP_CHANNEL


def load_prepared_dataset(path=None):
    if path:
        try:
            data = np.load(path)
            return (
                data["X_train"],
                data["Y_train"],
                data["X_val"],
                data["Y_val"],
            )
        except FileNotFoundError:
            logger.error(f"Directory \"{path}\" not found.")
    else:
        logger.error(f"Directory is None.")
        return None


def train_on_multiple_shots(source_dir,
                            filenames,
                            logger,
                            path_to_load="./data/dataset",
                            use_prepared=False):

    os.makedirs(path_to_load + "/dataset", exist_ok=True)

    loader = SHTLoader(source_dir)

    train_files, val_files = split_shots(filenames, val_ratio=0.2)

    logger.info(f"Train shots: {len(train_files)}")
    logger.info(f"Validation shots: {len(val_files)}")

    X_train_all, Y_train_all = [], []
    X_val_all, Y_val_all = [], []

    if use_prepared:
        X_train, Y_train, X_val, Y_val = load_prepared_dataset(path=path_to_load)

    else:
        for mode, file_list in [("train", train_files), ("val", val_files)]:

            logger.info(f"Processing {mode} shots...")

            for filename in file_list:

                shot = loader.load_shot(filename)

                if IP_CHANNEL not in shot.channel_names:
                    logger.warning("Ip channel not found → skip")
                    continue

                logger.info(f"--- {filename} {IP_CHANNEL} ---")

                ip_index = shot.channel_names.index(IP_CHANNEL)
                ip_signal = shot.signals[:, ip_index]

                plasma_interval = detect_plasma_interval(ip_signal, shot.time)

                if plasma_interval is None:
                    logger.warning("No plasma detected → skip")
                    continue

                t_start, t_end = plasma_interval
                mask = (shot.time >= t_start) & (shot.time <= t_end)

                signals_plasma = shot.signals[mask]
                time_plasma = shot.time[mask]

                dt = shot.dt

                logger.info(f"Plasma interval: {t_start:.4f} – {t_end:.4f} s")

                valid_channels = []

                for i, ch_name in enumerate(shot.channel_names):

                    if "SXR" not in ch_name:
                        continue

                    signal = signals_plasma[:, i]

                    is_saw, interval, period = detect_sawtooth_hybrid(signal, dt)
                    # is_saw, interval, period = detect_sawtooth_by_crashes(signal, dt)

                    if is_saw:
                        logger.info(f"{ch_name} → Sawtooth detected")
                        valid_channels.append((i, interval, period))
                    else:
                        logger.info(f"{ch_name} → No sawtooth")

                if not valid_channels:
                    logger.warning("No valid SXR channels → skip shot")
                    continue

                channel_idx, interval, period = valid_channels[0]

                start_idx, end_idx = interval

                signal = signals_plasma[start_idx:end_idx, channel_idx]

                if len(signal) < 500:
                    logger.warning("Sawtooth interval too short → skip")
                    continue

                signal = signal.reshape(-1, 1)

                signal = remove_mean(signal)
                signal = robust_scale(signal)

                deriv = compute_derivative(signal, dt)
                deriv = robust_scale(deriv)

                if period is None:
                    logger.warning("Period not estimated → skip")
                    continue

                # window_size = int((period / dt) * 0.5)

                GLOBAL_WINDOW_SIZE = 128
                window_size = GLOBAL_WINDOW_SIZE

                # GLOBAL_WINDOW_MS = 2.0
                # window_size = int((GLOBAL_WINDOW_MS * 1e-3) / dt)

                if window_size < 50:
                    window_size = 200

                logger.info(f"Estimated period: {period*1000:.2f} ms")
                logger.info(f"Window size: {window_size}")

                X, Y = create_windows(deriv, window_size)

                if len(X) == 0:
                    logger.warning("No windows created → skip")
                    continue

                logger.info(f"Windows created: {len(X)}")

                if mode == "train":
                    X_train_all.append(X)
                    Y_train_all.append(Y)
                else:
                    X_val_all.append(X)
                    Y_val_all.append(Y)

        if len(X_train_all) == 0:
            logger.error("No valid training data")
            return

        X_train = np.concatenate(X_train_all, axis=0)
        Y_train = np.concatenate(Y_train_all, axis=0)

        logger.info(f"Total train windows: {len(X_train)}")

        if len(X_val_all) > 0:
            X_val = np.concatenate(X_val_all, axis=0)
            Y_val = np.concatenate(Y_val_all, axis=0)

            logger.info(f"Total validation windows: {len(X_val)}")
        else:
            logger.warning("No validation data → training without validation")
            X_val, Y_val = None, None

        np.savez_compressed(
            path_to_load + "/dataset/sawtooth_dataset.npz",
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val
        )

        logger.info(f"Data saved to {os.path.abspath(path_to_load + "/dataset/sawtooth_dataset.npz")}")

        logger.info(f"Total train windows: {len(X_train)}")
        logger.info(f"Total validation windows: {len(X_val)}")

    train_loader = DataLoader(
        WindowDataset(X_train, Y_train),
        batch_size=256,
        shuffle=True
    )

    val_loader = DataLoader(
        WindowDataset(X_val, Y_val),
        batch_size=256,
        shuffle=False
    )

    input_dim = X_train.shape[2]

    model = TCNPredictor(input_dim=input_dim)

    train(
        model,
        train_loader,
        val_loader,
        logger,
        path_to_load=path_to_load + "/model_data/",
        epochs=50,
        lr=1e-3,
        patience=5
    )
