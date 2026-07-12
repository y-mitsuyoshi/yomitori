"""Tests for the logger utility."""

import logging
import tempfile
from pathlib import Path

from src.utils.logger import get_logger, setup_file_logging


def test_get_logger_returns_logger():
    """get_logger should return a configured logger."""
    logger = get_logger("test_module_1")
    assert isinstance(logger, logging.Logger)
    assert logger.handlers
    assert logger.propagate is False


def test_get_logger_reuses_existing():
    """get_logger should not add duplicate handlers."""
    logger1 = get_logger("test_module_2")
    handler_count = len(logger1.handlers)
    logger2 = get_logger("test_module_2")
    assert len(logger2.handlers) == handler_count


def test_setup_file_logging(tmp_path):
    """setup_file_logging should create a file handler and write to it."""
    log_file = tmp_path / "test.log"
    handler = setup_file_logging(str(log_file), level=logging.DEBUG)
    assert handler is not None
    logger = logging.getLogger("test_file_logging")
    logger.setLevel(logging.DEBUG)
    logger.info("Test message")
    handler.flush()
    assert log_file.exists()
    logging.getLogger().removeHandler(handler)
    handler.close()