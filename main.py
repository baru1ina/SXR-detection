from config.path import source_dir
from src.io.loader import SHTLoader
import argparse
from src.pipeline_train import train_on_multiple_shots
from src.detection.wavelet_pipeline import detect_on_shot as wavelet_detection
from src.detection.ml_pipeline import detect_on_shot as ml_detection


from src.logger import setup_logger


def main(mode=None, file=None, channel_name="SXR 50 mkm", logger=None):

    if mode is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", choices=["train", "detect"], required=True)
        parser.add_argument("--file", type=str, help="Filename for detection")
        args = parser.parse_args()
        mode = args.mode
        file = args.file

    loader = SHTLoader(source_dir)

    if mode == "train":
        print("TRAIN MODE")
        filenames = loader.list_files()
        print(f"Found {len(filenames)} shots")
        train_on_multiple_shots(source_dir, filenames, logger, path_to_load="./data")

    elif mode == "detect":
        if file is None:
            raise ValueError("Please specify file for detect mode")
        print("DETECT MODE")
        wavelet_detection(source_dir, file, logger, path_to_load="./data", channel_name=channel_name)


if __name__ == "__main__":
    logger = setup_logger()

    # Для запуска через терминал с аргументами
    # python script.py --mode train
    # python script.py --mode detect --file sht46358.SHT


    main(mode="detect", file="sht39627.SHT", logger=logger, channel_name="SXR 15 мкм")
    print("\n\n\n")
    # main(mode="detect", file="sht45898.SHT", logger=logger, channel_name="SXR 127 мкм")
    # print("\n\n\n")
    # main(mode="detect", file="sht42465.SHT", logger=logger, channel_name="SXR 80 mkm")
    # print("\n\n\n")
    # main(mode="detect", file="sht43770.SHT", logger=logger, channel_name="SXR 15 мкм")
    # print("\n\n\n")
    # main(mode="detect", file="sht43770.SHT", logger=logger, channel_name="SXR 127 мкм")
    # print("\n\n\n")
    # main(mode="detect", file="sht46358.SHT", logger=logger, channel_name="SXR 15 мкм")
    # print("\n\n\n")
    # main(mode="detect", file="sht45898.SHT", logger=logger)
    # print("\n\n\n")
    # main(mode="detect", file="sht45898.SHT", logger=logger, channel_name="SXR 15 мкм")
    # print("\n\n\n")
    # main(mode="detect", file="sht46358.SHT", logger=logger)
    # print("\n\n\n")
    # main(mode="detect", file="sht42465.SHT", logger=logger)
    # print("\n\n\n")
    # main(mode="detect", file="sht42465.SHT", logger=logger, channel_name="SXR 15 мкм")
    # print("\n\n\n")
    # main(mode="detect", file="sht46358.SHT", logger=logger, channel_name="SXR 80 mkm")
    # print("\n\n\n")
    # main(mode="detect", file="sht46345.SHT", logger=logger, channel_name="SXR 127 мкм")
    # print("\n\n\n")
    # main(mode="detect", file="sht44335.SHT", logger=logger, channel_name="SXR 127 мкм")
    # print("\n\n\n")
    # main(mode="detect", file="sht44335.SHT", logger=logger)
    # print("\n\n\n")
    # main(mode="detect", file="sht44335.SHT", logger=logger, channel_name="SXR 15 мкм")
    # print("\n\n\n")
    # main(mode="detect", file="sht38596.SHT", logger=logger, channel_name="SXR 15 мкм")
    # print("\n\n\n")
    # main(mode="detect", file="sht37622.SHT", logger=logger, channel_name="SXR 50 mkm")
    # print("\n\n\n")
    # main(mode="detect", file="sht37622.SHT", logger=logger, channel_name="SXR 15 мкм")
    # print("\n\n\n")
    # # main(mode="detect", file="sht46345.SHT", logger=logger)

    # main(mode="detect", file="sht46351.SHT", logger=logger, channel_name="SXR 127 мкм")
    # main(mode="detect", file="sht46351.SHT", logger=logger, channel_name="SXR 50 mkm")
    # main(mode="detect", file="sht46358.SHT", logger=logger, channel_name="SXR 15 мкм")
    # main(mode="detect", file="sht37622.SHT", logger=logger, channel_name="SXR 15 мкм")
    # main(mode="detect", file="sht37804.SHT", logger=logger, channel_name="SXR 15 мкм")
    # main(mode="detect", file="sht37804.SHT", logger=logger, channel_name="SXR 50 mkm")

    pass