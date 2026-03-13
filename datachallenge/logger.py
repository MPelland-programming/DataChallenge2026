import logging
import sys
from datachallenge.config import settings


def get_logger(name: str = "datachallenge") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)
    logger.addHandler(stderr_handler)

    if settings.log_file:
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.setLevel(settings.log_level.upper())
    return logger


logger = get_logger()
