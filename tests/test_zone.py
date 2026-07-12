"""Tests for Zone and DocumentType base classes."""

import numpy as np

from src.document_types.base import DocumentType, ValidationRule, Zone


def test_zone_contains_inside():
    """Zone.contains should return True for a point inside."""
    zone = Zone("f", "label", 0.0, 0.0, 0.5, 0.5)
    assert zone.contains(0.25, 0.25) is True


def test_zone_contains_boundary():
    """Zone.contains should return True on the boundary."""
    zone = Zone("f", "label", 0.0, 0.0, 0.5, 0.5)
    assert zone.contains(0.0, 0.0) is True
    assert zone.contains(0.5, 0.5) is True


def test_zone_contains_outside():
    """Zone.contains should return False for a point outside."""
    zone = Zone("f", "label", 0.0, 0.0, 0.5, 0.5)
    assert zone.contains(0.6, 0.6) is False


def test_zone_distance_inside():
    """Zone.distance_to should return 0.0 for a point inside."""
    zone = Zone("f", "label", 0.0, 0.0, 0.5, 0.5)
    assert zone.distance_to(0.25, 0.25) == 0.0


def test_zone_distance_x_outside():
    """Zone.distance_to should return x-distance when only x is outside."""
    zone = Zone("f", "label", 0.0, 0.0, 0.5, 0.5)
    dist = zone.distance_to(0.6, 0.25)
    assert abs(dist - 0.1) < 1e-6


def test_zone_distance_y_outside():
    """Zone.distance_to should return y-distance when only y is outside."""
    zone = Zone("f", "label", 0.0, 0.0, 0.5, 0.5)
    dist = zone.distance_to(0.25, 0.6)
    assert abs(dist - 0.1) < 1e-6


def test_zone_distance_both_outside():
    """Zone.distance_to should return Euclidean distance when both are outside."""
    zone = Zone("f", "label", 0.0, 0.0, 0.5, 0.5)
    dist = zone.distance_to(0.6, 0.6)
    expected = (0.1 ** 2 + 0.1 ** 2) ** 0.5
    assert abs(dist - expected) < 1e-6


def test_zone_defaults():
    """Zone optional fields should have correct defaults."""
    zone = Zone("f", "label", 0.0, 0.0, 0.5, 0.5)
    assert zone.label_remove is False
    assert zone.date_format is None
    assert zone.whitelist is None
    assert zone.normalize is None


def test_validation_rule_defaults():
    """ValidationRule should have correct defaults."""
    rule = ValidationRule()
    assert rule.pattern is None
    assert rule.check_digit is False
    assert rule.description == ""
    assert rule.required is True


def test_check_aspect_ratio_within_tolerance():
    """_check_aspect_ratio should return 1.0 within tolerance."""
    img = np.ones((100, 158, 3), dtype=np.uint8)
    score = DocumentType._check_aspect_ratio(img, expected=1.58, tolerance=0.05)
    assert score == 1.0


def test_check_aspect_ratio_outside_tolerance():
    """_check_aspect_ratio should return reduced score outside tolerance."""
    img = np.ones((100, 100, 3), dtype=np.uint8)
    score = DocumentType._check_aspect_ratio(img, expected=1.58, tolerance=0.05)
    assert 0.0 < score < 1.0


def test_check_aspect_ratio_clamped_to_zero():
    """_check_aspect_ratio should return very low score for extremely different ratios."""
    img = np.ones((1000, 1, 3), dtype=np.uint8)
    score = DocumentType._check_aspect_ratio(img, expected=1.58, tolerance=0.01)
    assert score < 0.01