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


def setup_logger(log_to_file=False, log_file_path="sawtooth.log"):
    logger = logging.getLogger("sawtooth")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        "%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        file_handler = logging.FileHandler("./logs/" + log_file_path, encoding='utf-8')
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S"  # Полная дата для файла
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    def newline(n: int = 1):
        for _ in range(n):
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.stream.write("\n")
                    handler.flush()
                elif isinstance(handler, logging.FileHandler):
                    handler.stream.write("\n")
                    handler.flush()

    logger.newline = newline

    return logger