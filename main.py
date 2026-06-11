from src.detection.ml.train.tcn_pipeline import train_on_multiple_shots

import argparse
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from config.path import SXR_CHANNELS, source_dir
from src.detection.cpd_pipeline import detect_on_shot as cpd_detection
from src.detection.posr_pipeline import detect_on_shot as posr_detection
from src.detection.wavelet_pipeline import detect_on_shot as wavelet_detection
from src.io.loader import SHTLoader
from src.logger import setup_logger

DetectionFn = Callable[..., object]

DETECTION_METHODS: Dict[str, DetectionFn] = {
    "wavelet": wavelet_detection,
    "cpd_features": cpd_detection,
    "cpd": cpd_detection,
    "wavelet_posr": posr_detection,
}

def main(
    mode: str,
    method: str = "wavelet",
    files_channels: Optional[List[Tuple[str, Optional[str]]]] = None,
    log_to_file: bool = True,
    log_file_path: Optional[str] = None,
    wt_thresholds: Optional[List[float]] = None,
    multichannel: bool = False,
    channels: Optional[List[str]] = None,
    min_channels: int = 2,
    coincidence_window: float = 0.3e-3,
    downsample: int = 10,
    plot: bool = True,
    debug: bool = False,
    penalty: float = 3.0,
    cpd_model: str = "rbf",
    wavelet_name: str = "mexh",
    cpd_score_threshold: float = 3.0,
    posr_sigma: Optional[float] = None,
    posr_threshold: float = 6.0,
    posr_score_threshold: float = 2.5,
    wavelet_use_local_period_nms=True,
):
    if log_file_path is None:
        log_file_path = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    logger = setup_logger(log_to_file=log_to_file, log_file_path=log_file_path)
    logger.info(f"Starting program in {mode} mode")

    loader = SHTLoader(source_dir, logger=logger)

    if mode == "train":
        filenames = loader.list_files()
        logger.info(f"Found {len(filenames)} shots")
        train_on_multiple_shots(source_dir, filenames, logger, path_to_load="./data")
        return

    if mode != "detect":
        raise ValueError(f"Unsupported mode: {mode}")

    method = method.lower()
    if method not in DETECTION_METHODS:
        raise ValueError(
            f"Unknown detection method: {method}. "
            f"Available: {sorted(DETECTION_METHODS)}"
        )

    if not files_channels:
        logger.error("No files for detection")
        logger.newline()
        raise ValueError("No files for detection")

    detector_fn = DETECTION_METHODS[method]
    wt_thresholds = wt_thresholds or [1.0] * len(files_channels)
    if len(wt_thresholds) < len(files_channels):
        wt_thresholds = wt_thresholds + [1.0] * (len(files_channels) - len(wt_thresholds))

    logger.info(f"Detection method: {method}")
    logger.info(f"Multichannel: {multichannel}")

    for i, (filename, channel_name) in enumerate(files_channels):
        selected_channel = channel_name or "SXR 50 mkm"

        try:
            common_kwargs = dict(
                source_dir=source_dir,
                filename=filename,
                logger=logger,
                path_to_load="./data",
                channel_name=selected_channel,
            )

            if method == "wavelet":
                detector_fn(
                    **common_kwargs,
                    wt_threshold=wt_thresholds[i],
                    downsample=downsample,
                    multichannel=multichannel,
                    channels=channels,
                    min_channels=min_channels,
                    coincidence_window=coincidence_window,
                    plot=plot,
                    debug=debug,
                    wavelet_use_local_period_nms=wavelet_use_local_period_nms,
                    wavelet_name=wavelet_name,
                )

            elif method in {"cpd", "cpd_features"}:
                detector_fn(
                    **common_kwargs,
                    penalty=penalty,
                    model=cpd_model,
                    score_threshold=cpd_score_threshold,
                    downsample=downsample,
                    multichannel=multichannel,
                    channels=channels,
                    min_channels=min_channels,
                    coincidence_window=coincidence_window,
                    plot=plot,
                    debug=debug,
                    use_features=True if method=="cpd_features" else False,
                )

            elif method in {"wavelet_posr"}:
                detector_fn(
                    **common_kwargs,
                    sigma=posr_sigma,
                    threshold=posr_threshold,
                    score_threshold=posr_score_threshold,
                    downsample=downsample,
                    multichannel=multichannel,
                    channels=channels,
                    min_channels=min_channels,
                    coincidence_window=coincidence_window,
                    plot=plot,
                    debug=debug,
                )

            logger.newline()

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            logger.newline()



def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Sawtooth crash detection runner"
    )
    parser.add_argument("--mode", choices=["train", "detect"], required=True)
    parser.add_argument(
        "--method",
        choices=sorted(DETECTION_METHODS.keys()),
        default="wavelet",
        help="Detection method for mode=detect",
    )

    #TODO: добавить сда  парсинг аргументов wavelet_raw_peak_distance, wavelet_distance_factor, crash_time_min

    parser.add_argument("--file", nargs="+", help="SHT filenames for detection")
    parser.add_argument("--ch", nargs="+", help="Reference channel names, one per file")
    parser.add_argument("--wt_threshold", type=float, nargs="+", help="Wavelet energy profile threshold")

    parser.add_argument("--multichannel", action="store_true", help="Run detector on several SXR channels and vote")
    parser.add_argument("--channels", nargs="+", default=None, help="SXR channels for multichannel mode")
    parser.add_argument("--min_channels", type=int, default=2)
    parser.add_argument("--coincidence_window", type=float, default=0.3e-3, help="Voting window in seconds")
    parser.add_argument("--downsample", type=int, default=10)
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--penalty", type=float, default=3.0, help="CPD penalty")
    parser.add_argument("--cpd_model", default="rbf", help="CPD model/kernel: rbf, linear, cosine, l2, ...")
    parser.add_argument("--cpd_score_threshold", type=float, default=3.0)

    parser.add_argument("--posr_sigma", type=float, default=None, help="wavelet posr sigma in seconds")
    parser.add_argument("--posr_threshold", type=float, default=6.0, help="POSR threshold")
    parser.add_argument("--posr_score_threshold", type=float, default=2.5)


    parser.add_argument("--model_path", default=None, help="Path to trained feature-ML .joblib model")
    parser.add_argument("--probability_threshold", type=float, default=0.5)

    args = parser.parse_args()

    files_channels: List[Tuple[str, Optional[str]]] = []
    if args.mode == "detect":
        if args.file:
            ch_list = args.ch or []
            for i, filename in enumerate(args.file):
                channel_name = ch_list[i] if i < len(ch_list) else None
                files_channels.append((filename, channel_name))

    wt_thresholds = args.wt_threshold or []
    channels = args.channels or SXR_CHANNELS

    return args, files_channels, wt_thresholds, channels


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        args, files_channels, wt_thresholds, channels = parse_cli_args()
        main(
            mode=args.mode,
            method=args.method,
            files_channels=files_channels,
            wt_thresholds=wt_thresholds,
            multichannel=args.multichannel,
            channels=channels,
            min_channels=args.min_channels,
            coincidence_window=args.coincidence_window,
            downsample=args.downsample,
            plot=not args.no_plot,
            debug=args.debug,
            penalty=args.penalty,
            cpd_model=args.cpd_model,
            cpd_score_threshold=args.cpd_score_threshold,
            posr_sigma=args.posr_sigma,
            posr_threshold=args.posr_threshold,
            posr_score_threshold=args.posr_score_threshold,
        )
    else:
        test_files_channels = [
            # ("sht46358.SHT", "SXR 15 мкм"),
            # ("sht39627.SHT", "SXR 15 мкм"),
            # ("sht38296.SHT", "SXR 50 mkm"),
            ("sht41025.SHT", "SXR 50 mkm"),
            ("sht41105.SHT", "SXR 50 mkm"),
            ("sht42465.SHT", "SXR 50 mkm"),
            ("sht43043.SHT", "SXR 50 mkm"),
            ("sht44335.SHT", "SXR 50 mkm"),
            ("sht44428.SHT", "SXR 50 mkm"),
        ]

        # test_files_channels = [
            # ("sht45898.SHT", "SXR 50 mkm"),
            # ("sht46358.SHT", "SXR 50 mkm"),
            # ("sht45898.SHT", "SXR 15 мкм"),
            # ("sht45898.SHT", "SXR 127 мкм"),
        # ]

        # test_files_channels = [
        #     ("sht43770.SHT", "SXR 50 mkm"),
        #     ("sht43838.SHT", "SXR 50 mkm"),
        # ]

        # test_files_channels = [
        #     # ("sht37804.SHT", "SXR 15 мкм"),
        #     # ("sht38596.SHT", "SXR 50 mkm"),
        #     ("sht39499.SHT", "SXR 15 мкм"),
        # ]

        # wt_thresholds = [2.0, 2.0]
        # wt_thresholds = [1.5, 1, 2]

        #TODO: сделать так, чтобы учитывалась периодическая структура.
        # Т.е. если на участке/в сигнале всего 1 срыв, то это не пила (!)

        # main(mode="detect", files_channels=test_files_channels, log_to_file=True)
        main(
            mode="detect",
            # method="cpd",
            # method="cpd_features",
            # method="wavelet_posr",
            files_channels=test_files_channels,
            # wt_thresholds=wt_thresholds,
            multichannel=False,
            # multichannel=True,
            # channels=SXR_CHANNELS,
            debug=False,
            # wavelet_name="mexh",
        )

        # main(
        #     mode="detect",
        #     # method="cpd",
        #     # method="cpd_features",
        #     # method="wavelet_posr",
        #     files_channels=test_files_channels,
        #     # wt_thresholds=wt_thresholds,
        #     multichannel=False,
        #     # multichannel=True,
        #     # channels=SXR_CHANNELS,
        #     debug=True,
        #     # wavelet_name="gaus1",
        # )
        #
        # main(
        #     mode="detect",
        #     # method="cpd",
        #     # method="cpd_features",
        #     method="wavelet_posr",
        #     files_channels=test_files_channels,
        #     wt_thresholds=wt_thresholds,
        #     multichannel=False,
        #     debug=True
        # )
        #
        # main(
        #     mode="detect",
        #     method="cpd",
        #     # method="cpd_features",
        #     # method="wavelet_posr",
        #     files_channels=test_files_channels,
        #     multichannel=False,
        #     debug=True
        # )
        #
        # main(
        #     mode="detect",
        #     # method="cpd",
        #     method="cpd_features",
        #     # method="wavelet_posr",
        #     files_channels=test_files_channels,
        #     multichannel=False,
        #     debug=True
        # )
