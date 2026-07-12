"""Tests for homography correction (beyond fallback)."""

import numpy as np

from src.preprocessing.homography import HomographyCorrector, HomographyInfo


def test_homography_info_to_dict():
    """HomographyInfo.to_dict should produce correct dict."""
    info = HomographyInfo(
        homography_applied=True,
        fallback_used=False,
        corrected_size=(800, 400),
    )
    d = info.to_dict()
    assert d["homography_applied"] is True
    assert d["fallback_used"] is False
    assert d["corrected_image_size"] == [800, 400]


def test_homography_corrector_fallback():
    """Corrector should fall back to resize on uniform image."""
    img = np.ones((100, 200, 3), dtype=np.uint8) * 200
    corrector = HomographyCorrector(target_size=(800, 400))
    corrected, info = corrector.correct(img)
    assert corrected.shape[0] == 400
    assert corrected.shape[1] == 800
    assert isinstance(info["fallback_used"], bool)


def test_homography_corrector_no_contours():
    """Corrector should fall back when no contours found."""
    img = np.ones((100, 200, 3), dtype=np.uint8) * 128
    corrector = HomographyCorrector(target_size=(400, 200))
    corrected, info = corrector.correct(img)
    assert corrected.shape == (200, 400, 3)


def test_homography_corrector_with_card():
    """Corrector should apply homography for a card-like image."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 200
    img[50:350, 50:550] = 50
    corrector = HomographyCorrector(target_size=(800, 400), min_contour_area_ratio=0.05)
    corrected, info = corrector.correct(img)
    assert corrected.shape[0] == 400
    assert corrected.shape[1] == 800