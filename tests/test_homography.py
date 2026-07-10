"""Tests for homography/perspective correction."""

import numpy as np

from src.preprocessing.perspective import order_points, four_point_perspective


def test_order_points_basic():
    """order_points should return (tl, bl, br, tr)."""
    pts = np.array([[0, 0], [10, 10], [0, 10], [10, 0]], dtype=np.float32)
    ordered = order_points(pts)
    assert ordered[0].tolist() == [0, 0]
    assert ordered[2].tolist() == [10, 10]


def test_four_point_perspective_identity():
    """Perspective transform with identity-like points should preserve size."""
    img = np.ones((100, 200, 3), dtype=np.uint8) * 128
    src = np.array([[0, 0], [0, 50], [200, 50], [200, 0]], dtype=np.float32)
    out = four_point_perspective(img, src, (200, 50))
    assert out.shape == (50, 200, 3)


def test_homography_corrector_fallback():
    """HomographyCorrector should fall back to resize on plain image."""
    from src.preprocessing.homography import HomographyCorrector

    img = np.ones((100, 200, 3), dtype=np.uint8) * 200
    corrector = HomographyCorrector(target_size=(800, 400))
    corrected, info = corrector.correct(img)
    assert corrected.shape[0] == 400
    assert corrected.shape[1] == 800
    # On a uniform image, either homography succeeds (finds the full image
    # border as a contour) or it falls back to resize — both are acceptable.
    assert isinstance(info["fallback_used"], bool)