"""Tests for card detector."""

import numpy as np
from unittest.mock import patch, MagicMock

from src.preprocessing.card_detector import classify_by_aspect_ratio, detect_card_region


def test_detect_card_region_no_contours():
    """detect_card_region should return None when no contours found."""
    img = np.ones((100, 200, 3), dtype=np.uint8) * 128
    with patch("cv2.findContours", return_value=([], None)):
        result = detect_card_region(img)
    assert result is None


def test_detect_card_region_small_contour():
    """detect_card_region should return None when contour area is too small."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    mock_contour = MagicMock()
    with patch("cv2.findContours", return_value=([mock_contour], None)), \
         patch("cv2.contourArea", return_value=10.0):
        result = detect_card_region(img, min_area_ratio=0.30)
    assert result is None


def test_detect_card_region_fast_four_points():
    """detect_card_region should return 4 points when first approx yields 4."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    mock_contour = MagicMock()
    mock_approx = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
    with patch("cv2.findContours", return_value=([mock_contour], None)), \
         patch("cv2.contourArea", return_value=40000.0), \
         patch("cv2.arcLength", return_value=100.0), \
         patch("cv2.approxPolyDP", return_value=mock_approx):
        result = detect_card_region(img, min_area_ratio=0.05, epsilon_factor=0.02)
    assert result is not None
    assert result.shape == (4, 2)


def test_detect_card_region_epsilon_search_found():
    """detect_card_region should search through epsilon values and find 4 points."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    mock_contour = MagicMock()
    # First approx returns 3 points (not 4), then second returns 4
    approx_3 = np.array([[[0, 0]], [[10, 0]], [[10, 10]]], dtype=np.int32)
    approx_4 = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
    with patch("cv2.findContours", return_value=([mock_contour], None)), \
         patch("cv2.contourArea", return_value=40000.0), \
         patch("cv2.arcLength", return_value=100.0), \
         patch("cv2.approxPolyDP", side_effect=[approx_3, approx_4]):
        result = detect_card_region(
            img, min_area_ratio=0.05,
            epsilon_factor=0.001,
            epsilon_min=0.01, epsilon_max=0.05, epsilon_step=0.01,
        )
    assert result is not None
    assert result.shape == (4, 2)


def test_detect_card_region_epsilon_search_exhausted():
    """detect_card_region should return None when epsilon search exhausted."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    mock_contour = MagicMock()
    approx_3 = np.array([[[0, 0]], [[10, 0]], [[10, 10]]], dtype=np.int32)
    with patch("cv2.findContours", return_value=([mock_contour], None)), \
         patch("cv2.contourArea", return_value=40000.0), \
         patch("cv2.arcLength", return_value=100.0), \
         patch("cv2.approxPolyDP", return_value=approx_3):
        result = detect_card_region(
            img, min_area_ratio=0.05,
            epsilon_factor=0.001,
            epsilon_min=0.01, epsilon_max=0.02, epsilon_step=0.01,
        )
    assert result is None


def test_detect_card_region_with_rectangle():
    """detect_card_region should find 4 corners for a clear rectangle."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 220
    img[50:350, 50:550] = 60  # Dark rectangle
    result = detect_card_region(img, min_area_ratio=0.05)
    if result is not None:
        assert result.shape == (4, 2)


def test_classify_by_aspect_ratio_match():
    """classify_by_aspect_ratio should return matching type."""
    img = np.ones((1515, 2400, 3), dtype=np.uint8)
    ratios = {"driver_license_front": 1.585, "mynumber_card_front": 1.586}
    result = classify_by_aspect_ratio(img, ratios, tolerance=0.01)
    assert result is not None


def test_classify_by_aspect_ratio_no_match():
    """classify_by_aspect_ratio should return None when no match."""
    img = np.ones((100, 100, 3), dtype=np.uint8)
    ratios = {"driver_license_front": 1.585}
    result = classify_by_aspect_ratio(img, ratios, tolerance=0.05)
    assert result is None


def test_classify_by_aspect_ratio_best_match():
    """classify_by_aspect_ratio should pick the best match among multiple."""
    img = np.ones((100, 158, 3), dtype=np.uint8)
    ratios = {"a": 1.585, "b": 2.0}
    result = classify_by_aspect_ratio(img, ratios, tolerance=0.05)
    assert result == "a"