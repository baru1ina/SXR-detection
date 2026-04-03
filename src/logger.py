import logging
import sys

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[91m',
        'INFO': '\033[0m',
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)

        formatted = super().format(record)

        return f"{color}{formatted}{self.RESET}"


def setup_logger():
    logger = logging.getLogger("sawtooth")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = ColoredFormatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        "%H:%M:%S"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger