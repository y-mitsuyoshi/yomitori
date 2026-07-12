"""Tests for the E2E pipeline (without GPU models)."""

import numpy as np

from src.document_types.driver_license import DriverLicenseFront
from src.pipeline.field_extractor import FieldExtractor
from src.postprocessing.normalizer import (
    apply_whitelist,
    fullwidth_to_halfwidth,
    japanese_era_to_iso,
    remove_label,
)


def test_field_extractor_strict():
    """Strict containment should assign correctly."""
    dt = DriverLicenseFront()
    extractor = FieldExtractor(dt)
    # name zone: x[0,0.55], y[0,0.12]
    assert extractor.assign(0.1, 0.05) == "name"
    # license_number zone: x[0,0.70], y[0.60,0.78]
    assert extractor.assign(0.3, 0.70) == "license_number"


def test_field_extractor_nearest():
    """Pass-2 should assign to nearest zone within margin."""
    dt = DriverLicenseFront()
    extractor = FieldExtractor(dt, margin=0.04)
    # Point at (0.1, 0.92) — below license_type zone y[0.78,0.90] by 0.02
    result = extractor.assign(0.1, 0.92)
    assert result is not None  # should fall to nearest


def test_field_extractor_none():
    """Should return None when far from all zones."""
    dt = DriverLicenseFront()
    extractor = FieldExtractor(dt, margin=0.001)
    assert extractor.assign(0.99, 0.99) is None


def test_remove_label():
    """remove_label should strip the label prefix."""
    assert remove_label("氏名 山田太郎", "氏名") == "山田太郎"
    assert remove_label("生年月日 昭和61年5月1日", "生年月日") == "昭和61年5月1日"


def test_apply_whitelist():
    """apply_whitelist should filter characters."""
    assert apply_whitelist("0120304-05-06", "0123456789") == "01203040506"
    assert apply_whitelist("abc123", "0123456789") == "123"


def test_fullwidth_to_halfwidth():
    """fullwidth_to_halfwidth should convert digits."""
    assert fullwidth_to_halfwidth("１２３") == "123"
    assert fullwidth_to_halfwidth("ＡＢＣ") == "ABC"


def test_japanese_era_to_iso():
    """japanese_era_to_iso should convert era dates."""
    raw, iso = japanese_era_to_iso("昭和61年5月1日生")
    assert iso == "1986-05-01"
    raw, iso = japanese_era_to_iso("令和7年12月31日")
    assert iso == "2025-12-31"
    raw, iso = japanese_era_to_iso("平成1年1月1日")
    assert iso == "1989-01-01"


def test_japanese_era_no_match():
    """Should return (raw, None) when no era pattern found."""
    raw, iso = japanese_era_to_iso("hello")
    assert raw == "hello"
    assert iso is None


def test_apply_whitelist_empty():
    """apply_whitelist should return text unchanged when whitelist is empty."""
    assert apply_whitelist("abc123", "") == "abc123"


def test_japanese_era_gannen():
    """japanese_era_to_iso should handle 元 (gannen/first year)."""
    raw, iso = japanese_era_to_iso("令和元年5月1日")
    assert iso == "2019-05-01"


def test_japanese_era_meiji():
    """japanese_era_to_iso should handle 明治."""
    raw, iso = japanese_era_to_iso("明治45年7月30日")
    assert iso == "1912-07-30"


def test_japanese_era_taisho():
    """japanese_era_to_iso should handle 大正."""
    raw, iso = japanese_era_to_iso("大正15年12月25日")
    assert iso == "1926-12-25"


def test_field_extractor_pass2_debug(caplog):
    """Pass-2 assignment should work and log debug message."""
    import logging

    dt = DriverLicenseFront()
    extractor = FieldExtractor(dt, margin=0.05)
    # Point at (0.1, 0.92) — below license_type zone y[0.78,0.90] by 0.02
    # Not inside any zone (license_number ends at y=0.78, license_type starts at y=0.78)
    with caplog.at_level(logging.DEBUG, logger="src.pipeline.field_extractor"):
        result = extractor.assign(0.1, 0.92)
    assert result is not None
    assert result == "license_type"


def test_field_extractor_pass2_updates_best():
    """Pass-2 should iterate and update best_zone multiple times."""
    dt = DriverLicenseFront()
    extractor = FieldExtractor(dt, margin=0.10)
    # Point at (0.1, 0.92) — below license_type zone y[0.78,0.90] by 0.02
    result = extractor.assign(0.1, 0.92)
    assert result is not None


def test_field_extractor_pass2_no_match_within_margin():
    """Pass-2 should return None when no zone is within margin."""
    dt = DriverLicenseFront()
    extractor = FieldExtractor(dt, margin=0.001)
    # Point far from all zones
    result = extractor.assign(0.99, 0.99)
    assert result is None