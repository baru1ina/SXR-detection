from config.path import source_dir
from src.io.loader import SHTLoader
from src.pipeline_train import train_on_multiple_shots
from src.detection.wavelet_pipeline import detect_on_shot as wavelet_detection
from typing import List, Tuple, Optional
import argparse
from datetime import datetime

from src.logger import setup_logger


def main(mode: str,
         files_channels: Optional[List[Tuple[str, Optional[str]]]] = None,
         log_to_file: bool = True,
         log_file_path: Optional[str] = None):

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

        for filename, channel_name in files_channels:
            logger.newline()
            logger.info(f"Processing: {filename} (channel: {channel_name})")
            try:
                wavelet_detection(
                    source_dir,
                    filename,
                    logger,
                    path_to_load="./data",
                    channel_name=channel_name or "SXR 50 mkm"
                )
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")


def parse_cli_args() -> Tuple[str, List[Tuple[str, Optional[str]]]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "detect"], required=True)
    parser.add_argument("--file", nargs="+", help="filenames for detection")
    parser.add_argument("--ch", nargs="+", help="channel names")
    args = parser.parse_args()
    files_channels: List[Tuple[str, Optional[str]]] = []
    if args.mode == "detect" and args.file:
        ch_list = args.ch or []
        for i, filename in enumerate(args.file):
            channel_name = ch_list[i] if i < len(ch_list) else None
            files_channels.append((filename, channel_name))

    return args.mode, files_channels


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode, files_channels = parse_cli_args()
        main(mode=mode, files_channels=files_channels)
    else:
        test_files_channels = [
            ("sht46358.SHT", "SXR 15 мкм"),
            ("sht39627.SHT", "SXR 15 мкм"),
            ("sht45898.SHT", "SXR 127 мкм"),
        ]
        main(mode="detect", files_channels=test_files_channels)

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
