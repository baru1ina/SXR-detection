import logging
import sys


def setup_logger():
    logger = logging.getLogger("sawtooth")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        "%H:%M:%S"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger