"""Tests for DriverLicenseFront and MyNumberCardFront document types."""

import numpy as np

from src.document_types.driver_license import DriverLicenseFront
from src.document_types.mynumber_card import MyNumberCardFront


def test_driver_license_properties():
    """DriverLicenseFront should have correct properties."""
    dt = DriverLicenseFront()
    assert dt.document_type_id == "driver_license_front"
    assert dt.card_aspect_ratio == 1.585
    assert dt.normalized_size == (2400, 1512)
    assert len(dt.zones) == 8
    assert len(dt.validation_rules) == 3


def test_driver_license_detect_match():
    """detect should return high score for matching aspect ratio."""
    dt = DriverLicenseFront()
    img = np.ones((1515, 2400, 3), dtype=np.uint8)
    score = dt.detect(img)
    assert score == 1.0


def test_driver_license_detect_no_match():
    """detect should return low score for non-matching aspect ratio."""
    dt = DriverLicenseFront()
    img = np.ones((100, 100, 3), dtype=np.uint8)
    score = dt.detect(img)
    assert score < 1.0


def test_mynumber_card_properties():
    """MyNumberCardFront should have correct properties."""
    dt = MyNumberCardFront()
    assert dt.document_type_id == "mynumber_card_front"
    assert dt.card_aspect_ratio == 1.586
    assert dt.normalized_size == (2400, 1513)
    assert len(dt.zones) == 7
    assert len(dt.validation_rules) == 3


def test_mynumber_card_detect_match():
    """detect should return high score for matching aspect ratio."""
    dt = MyNumberCardFront()
    img = np.ones((1513, 2400, 3), dtype=np.uint8)
    score = dt.detect(img)
    assert score == 1.0


def test_mynumber_card_zones_have_whitelist():
    """mynumber zone should have a digit whitelist."""
    dt = MyNumberCardFront()
    mynumber_zone = dt.zones["mynumber"]
    assert mynumber_zone.whitelist == "0123456789"