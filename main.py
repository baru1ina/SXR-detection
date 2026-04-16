from config.path import source_dir
from src.io.loader import SHTLoader
from src.ml.train.tcn_pipeline import train_on_multiple_shots
from src.detection.wavelet_pipeline import detect_on_shot as wavelet_detection
from src.detection.cpd_pipeline import detect_on_shot as cpd_detection
from typing import List, Tuple, Optional
import argparse
from datetime import datetime

from src.logger import setup_logger


def main(mode: str,
         files_channels: Optional[List[Tuple[str, Optional[str]]]] = None,
         log_to_file: bool = True,
         log_file_path: Optional[str] = None,
         wt_thresholds: Optional[List[float]] = None):

    if log_file_path is None:
        log_file_path = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    logger = setup_logger(log_to_file=log_to_file, log_file_path=log_file_path)
    logger.info(f"Starting program in {mode} mode")

    loader = SHTLoader(source_dir, logger=logger)

    if mode == "train":
        filenames = loader.list_files()
        logger.info(f"Found {len(filenames)} shots")
        train_on_multiple_shots(source_dir, filenames, logger, path_to_load="./data")

    elif mode == "detect":
        if not files_channels:
            logger.error("No files for detection")
            raise ValueError("No files for detection")

        for i, (filename, channel_name) in enumerate(files_channels):
            logger.newline()
            logger.info(f"Processing: {filename} (channel: {channel_name})")
            try:
                # wavelet_detection(
                #     source_dir,
                #     filename,
                #     logger,
                #     path_to_load="./data",
                #     channel_name=channel_name or "SXR 50 mkm",
                #     wt_threshold=wt_thresholds[i]
                # )
                cpd_detection(
                    source_dir,
                    filename,
                    logger,
                    path_to_load="./data",
                    channel_name=channel_name or "SXR 50 mkm"
                )
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "detect"], required=True)
    parser.add_argument("--file", nargs="+", help="filenames for detection")
    parser.add_argument("--ch", nargs="+", help="channel names")
    parser.add_argument("--wt_threshold", type=int, nargs="+", help="wavelet energy profile threshold")
    args = parser.parse_args()
    files_channels: List[Tuple[str, Optional[str]]] = []
    wt_thresholds: List[float] = []
    if args.mode == "detect" and args.file:
        ch_list = args.ch or []
        wt_threshold_list = args.wt_threshold or []
        for i, filename in enumerate(args.file):
            channel_name = ch_list[i] if i < len(ch_list) else None
            wt_threshold = wt_threshold_list[i] if i < len(wt_threshold_list) else 1.0
            print(wt_threshold, type(wt_threshold))
            files_channels.append((filename, channel_name))
            wt_thresholds.append(wt_threshold)
    elif args.mode == "train" and args.file:
        pass

    return args.mode, files_channels, wt_thresholds


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode, files_channels, wt_thresholds = parse_cli_args()
        main(mode=mode, files_channels=files_channels, wt_thresholds=wt_thresholds)
    else:
        # test_files_channels = [
        #     ("sht46358.SHT", "SXR 15 мкм"),
        #     ("sht39627.SHT", "SXR 15 мкм"),
        #     ("sht45898.SHT", "SXR 127 мкм"),
        # ]

        test_files_channels = [
            ("sht46358.SHT", "SXR 50 mkm"),
            # ("sht39627.SHT", "SXR 15 мкм"),
            # ("sht45898.SHT", "SXR 127 мкм"),
        ]

        # wt_thresholds = [1.5, 1, 2]

        main(mode="detect", files_channels=test_files_channels, log_to_file=False)
        # main(mode="detect", files_channels=test_files_channels, wt_thresholds=wt_thresholds)

# if __name__ == "__main__":
#     logger = setup_logger(log_to_file=True, log_file_path=f"log_{datetime.now().date()} {datetime.now().hour}-{datetime.now().minute}-{datetime.now().second}.log")
#
#     main(mode="detect", file="sht46358.SHT", logger=logger, channel_name="SXR 15 мкм")
#     print("\n\n\n")
#     main(mode="detect", file="sht39627.SHT", logger=logger, channel_name="SXR 15 мкм")
#     print("\n\n\n")
#     main(mode="detect", file="sht45898.SHT", logger=logger, channel_name="SXR 127 мкм")
#     print("\n\n\n")
#     main(mode="detect", file="sht42465.SHT", logger=logger, channel_name="SXR 80 mkm")
#     print("\n\n\n")
#     main(mode="detect", file="sht43770.SHT", logger=logger, channel_name="SXR 15 мкм")
#     print("\n\n\n")
#     main(mode="detect", file="sht43770.SHT", logger=logger, channel_name="SXR 127 мкм")
#     print("\n\n\n")
#     main(mode="detect", file="sht46358.SHT", logger=logger, channel_name="SXR 15 мкм")
#     print("\n\n\n")
#     main(mode="detect", file="sht45898.SHT", logger=logger)
#     print("\n\n\n")
#     main(mode="detect", file="sht45898.SHT", logger=logger, channel_name="SXR 15 мкм")
#     print("\n\n\n")
#     main(mode="detect", file="sht46358.SHT", logger=logger)
#     print("\n\n\n")
#     main(mode="detect", file="sht42465.SHT", logger=logger)
#     print("\n\n\n")
#     main(mode="detect", file="sht42465.SHT", logger=logger, channel_name="SXR 15 мкм")
#     print("\n\n\n")
#     main(mode="detect", file="sht46358.SHT", logger=logger, channel_name="SXR 80 mkm")
#     print("\n\n\n")
#     main(mode="detect", file="sht46345.SHT", logger=logger, channel_name="SXR 127 мкм")
#     print("\n\n\n")
#     main(mode="detect", file="sht44335.SHT", logger=logger, channel_name="SXR 127 мкм")
#     print("\n\n\n")
#     main(mode="detect", file="sht44335.SHT", logger=logger)
#     print("\n\n\n")
#     main(mode="detect", file="sht44335.SHT", logger=logger, channel_name="SXR 15 мкм")
#     print("\n\n\n")
#     main(mode="detect", file="sht38596.SHT", logger=logger, channel_name="SXR 15 мкм")
#     print("\n\n\n")
#     main(mode="detect", file="sht37622.SHT", logger=logger, channel_name="SXR 50 mkm")
#     print("\n\n\n")
#     main(mode="detect", file="sht37622.SHT", logger=logger, channel_name="SXR 15 мкм")
#     print("\n\n\n")
#     main(mode="detect", file="sht46345.SHT", logger=logger)
#     print("\n\n\n")
#     main(mode="detect", file="sht46351.SHT", logger=logger, channel_name="SXR 127 мкм")
#     print("\n\n\n")
#     main(mode="detect", file="sht46351.SHT", logger=logger, channel_name="SXR 50 mkm")
#     print("\n\n\n")
#     main(mode="detect", file="sht46358.SHT", logger=logger, channel_name="SXR 15 мкм")
#     print("\n\n\n")
#     main(mode="detect", file="sht37622.SHT", logger=logger, channel_name="SXR 15 мкм")
#     print("\n\n\n")
#     main(mode="detect", file="sht37804.SHT", logger=logger, channel_name="SXR 15 мкм")
#     print("\n\n\n")
#     main(mode="detect", file="sht37804.SHT", logger=logger, channel_name="SXR 50 mkm")
