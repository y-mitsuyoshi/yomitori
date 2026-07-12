"""Tests for the recognition whitelist module."""

from src.recognition.whitelist import apply_blacklist, apply_whitelist


def test_apply_whitelist_basic():
    """apply_whitelist should keep only whitelisted chars."""
    assert apply_whitelist("abc123", "0123456789") == "123"


def test_apply_whitelist_empty_whitelist():
    """apply_whitelist should return text unchanged when whitelist is empty."""
    assert apply_whitelist("abc", "") == "abc"


def test_apply_blacklist_basic():
    """apply_blacklist should remove blacklisted chars."""
    assert apply_blacklist("abc123", "123") == "abc"


def test_apply_blacklist_empty_blacklist():
    """apply_blacklist should return text unchanged when blacklist is empty."""
    assert apply_blacklist("abc", "") == "abc"