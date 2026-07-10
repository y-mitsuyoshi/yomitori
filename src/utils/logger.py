"""Logger utility for the Yomitori OCR engine."""

import logging
import sys
from pathlib import Path

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger.

    Args:
        name: Logger name (usually __name__).
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def setup_file_logging(log_path: str | Path, level: int = logging.DEBUG) -> logging.Handler:
    """Add a file handler to the root logger.

    Args:
        log_path: Path to the log file.
        level: Logging level for the file handler.

    Returns:
        The file handler that was added.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DATE_FORMAT))
    logging.getLogger().addHandler(handler)
    return handler