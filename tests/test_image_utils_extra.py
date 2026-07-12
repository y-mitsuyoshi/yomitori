"""Tests for image_utils visualize_preprocessing and save_image error handling."""

import numpy as np
import pytest


def test_visualize_preprocessing_all_steps():
    """visualize_preprocessing should handle all image types."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend

    from src.utils.image_utils import visualize_preprocessing

    image = np.zeros((50, 100, 3), dtype=np.uint8)
    gray = np.zeros((50, 100), dtype=np.uint8)
    binary = np.zeros((50, 100), dtype=np.uint8)
    contours = np.zeros((50, 100, 3), dtype=np.uint8)
    corrected = np.zeros((50, 100, 3), dtype=np.uint8)

    # Should not raise
    visualize_preprocessing(image, gray, binary, contours, corrected)


def test_visualize_preprocessing_single_step():
    """visualize_preprocessing should work with just the original image."""
    import matplotlib
    matplotlib.use("Agg")

    from src.utils.image_utils import visualize_preprocessing

    image = np.zeros((50, 100, 3), dtype=np.uint8)
    visualize_preprocessing(image)


def test_visualize_preprocessing_gray_only():
    """visualize_preprocessing should handle grayscale-only case."""
    import matplotlib
    matplotlib.use("Agg")

    from src.utils.image_utils import visualize_preprocessing

    image = np.zeros((50, 100, 3), dtype=np.uint8)
    gray = np.zeros((50, 100), dtype=np.uint8)
    visualize_preprocessing(image, gray=gray)


def test_save_image_failure():
    """save_image should raise IOError on invalid path."""
    from src.utils.image_utils import save_image

    img = np.zeros((50, 100, 3), dtype=np.uint8)
    # Write to a path under a file (not a directory) to force failure
    with pytest.raises((IOError, Exception)):
        save_image(img, "/nonexistent_dir_xyz/deep/test.png")