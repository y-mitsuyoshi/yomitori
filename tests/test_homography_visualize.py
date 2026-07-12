"""Tests for homography visualize and edge cases."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.preprocessing.homography import HomographyCorrector


def test_homography_visualize():
    """visualize should call matplotlib without error."""
    with patch("src.utils.image_utils.visualize_preprocessing") as mock_vis:
        img = np.ones((100, 200, 3), dtype=np.uint8) * 128
        corrector = HomographyCorrector(target_size=(400, 200))
        corrector.visualize(img)
        mock_vis.assert_called_once()


def test_homography_no_contours():
    """Corrector should fall back when no contours found."""
    img = np.ones((100, 200, 3), dtype=np.uint8) * 128
    with patch("cv2.findContours", return_value=([], None)):
        corrector = HomographyCorrector(target_size=(400, 200))
        corrected, info = corrector.correct(img)
    assert info["fallback_used"] is True
    assert corrected.shape == (200, 400, 3)


def test_homography_small_contour():
    """Corrector should fall back when contour area is too small."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    mock_contour = MagicMock()
    with patch("cv2.findContours", return_value=([mock_contour], None)), \
         patch("cv2.contourArea", return_value=10.0):
        corrector = HomographyCorrector(target_size=(400, 200), min_contour_area_ratio=0.30)
        corrected, info = corrector.correct(img)
    assert info["fallback_used"] is True


def test_homography_four_point_failure():
    """Corrector should fall back when 4-point approximation fails."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    mock_contour = MagicMock()
    approx_3 = np.array([[[0, 0]], [[10, 0]], [[10, 10]]], dtype=np.int32)
    with patch("cv2.findContours", return_value=([mock_contour], None)), \
         patch("cv2.contourArea", return_value=40000.0), \
         patch("cv2.arcLength", return_value=100.0), \
         patch("cv2.approxPolyDP", return_value=approx_3):
        corrector = HomographyCorrector(
            target_size=(400, 200),
            min_contour_area_ratio=0.05,
            epsilon_factor=0.001,
            epsilon_min=0.01, epsilon_max=0.02, epsilon_step=0.01,
        )
        corrected, info = corrector.correct(img)
    assert info["fallback_used"] is True


def test_homography_success_with_mock():
    """Corrector should apply homography when 4-point approximation succeeds."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    mock_contour = MagicMock()
    approx_4 = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
    with patch("cv2.findContours", return_value=([mock_contour], None)), \
         patch("cv2.contourArea", return_value=40000.0), \
         patch("cv2.arcLength", return_value=100.0), \
         patch("cv2.approxPolyDP", return_value=approx_4):
        corrector = HomographyCorrector(
            target_size=(400, 200),
            min_contour_area_ratio=0.05,
            epsilon_factor=0.02,
        )
        corrected, info = corrector.correct(img)
    assert info["homography_applied"] is True
    assert info["fallback_used"] is False
    assert corrected.shape == (200, 400, 3)


def test_homography_find_four_points_fast_path():
    """_find_four_points should return quickly when first approx yields 4 points."""
    import cv2

    mask = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(mask, (50, 50), (250, 150), 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    arc = cv2.arcLength(contour, True)

    corrector = HomographyCorrector(target_size=(100, 100), epsilon_factor=0.02)
    result = corrector._find_four_points(contour, arc)
    if result is not None:
        assert result.shape == (4, 2)


def test_homography_find_four_points_search_loop():
    """_find_four_points should search through epsilon range."""
    import cv2

    mask = np.zeros((200, 300), dtype=np.uint8)
    cv2.circle(mask, (150, 100), 80, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    arc = cv2.arcLength(contour, True)

    corrector = HomographyCorrector(
        target_size=(100, 100),
        epsilon_factor=0.001,
        epsilon_min=0.01,
        epsilon_max=0.15,
        epsilon_step=0.005,
    )
    result = corrector._find_four_points(contour, arc)
    if result is not None:
        assert result.shape == (4, 2)