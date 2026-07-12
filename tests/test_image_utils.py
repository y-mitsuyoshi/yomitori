"""Tests for image utilities."""

from unittest.mock import patch

import numpy as np
import pytest

from src.utils.image_utils import (
    decode_base64_image,
    decode_image,
    draw_bboxes,
    encode_image,
    resize_image,
    save_image,
)


def test_decode_image_valid():
    """decode_image should decode valid PNG bytes."""
    import cv2

    img = np.zeros((50, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    decoded = decode_image(buf.tobytes())
    assert decoded.shape == (50, 100, 3)


def test_decode_image_invalid():
    """decode_image should raise ValueError on invalid bytes."""
    with pytest.raises(ValueError, match="Failed to decode"):
        decode_image(b"not an image")


def test_decode_base64_image_with_prefix():
    """decode_base64_image should handle data URI prefix."""
    import base64
    import cv2

    img = np.zeros((50, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    b64 = base64.b64encode(buf.tobytes()).decode()
    decoded = decode_base64_image(f"data:image/png;base64,{b64}")
    assert decoded.shape == (50, 100, 3)


def test_decode_base64_image_without_prefix():
    """decode_base64_image should handle plain base64."""
    import base64
    import cv2

    img = np.zeros((50, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    b64 = base64.b64encode(buf.tobytes()).decode()
    decoded = decode_base64_image(b64)
    assert decoded.shape == (50, 100, 3)


def test_encode_image():
    """encode_image should return bytes."""
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    data = encode_image(img, ".png")
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_encode_image_invalid():
    """encode_image should raise on failure."""
    with patch("cv2.imencode", return_value=(False, None)):
        with pytest.raises(ValueError, match="Failed to encode"):
            encode_image(np.zeros((10, 10, 3), dtype=np.uint8), ".png")


def test_resize_image():
    """resize_image should resize to target."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    resized = resize_image(img, (50, 25))
    assert resized.shape == (25, 50, 3)


def test_draw_bboxes():
    """draw_bboxes should draw boxes on a copy."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    bboxes = [[(10, 10), (50, 10), (50, 50), (10, 50)]]
    result = draw_bboxes(img, bboxes)
    assert result.shape == img.shape
    # Original should be unchanged
    assert np.array_equal(img, np.zeros((100, 200, 3), dtype=np.uint8))


def test_save_image(tmp_path):
    """save_image should write to disk."""
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    path = str(tmp_path / "test.png")
    save_image(img, path)
    import os
    assert os.path.exists(path)