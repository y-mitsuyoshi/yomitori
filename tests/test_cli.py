"""Tests for the CLI module."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np


def test_cli_no_command(capsys):
    """CLI with no command should print help and return 1."""
    from src.cli import main

    old_argv = sys.argv
    sys.argv = ["yomitori"]
    try:
        ret = main()
        assert ret == 1
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "help" in captured.out.lower()
    finally:
        sys.argv = old_argv


def test_cli_build_registry():
    """build_registry should include all registered document types."""
    from src.cli import build_registry

    registry = build_registry()
    type_ids = registry.list_types()
    assert "driver_license_front" in type_ids
    assert "mynumber_card_front" in type_ids