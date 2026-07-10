"""Image binarization using Otsu's method with adaptive fallbacks."""

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def otsu_binarization(gray: np.ndarray) -> np.ndarray:
    """Apply Otsu's threshold to a grayscale image.

    Args:
        gray: Grayscale image.

    Returns:
        Binary image (0/255).
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def adaptive_binarization(
    gray: np.ndarray,
    block_size: int = 31,
    c: int = 5,
) -> np.ndarray:
    """Adaptive Gaussian threshold (useful when Otsu fails on uneven lighting).

    Args:
        gray: Grayscale image.
        block_size: Neighborhood size (must be odd).
        c: Constant subtracted from the mean.

    Returns:
        Binary image.
    """
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def dynamic_binarization(gray: np.ndarray) -> np.ndarray:
    """Pick Otsu or adaptive based on histogram analysis.

    If the image is heavily skewed (bright background dominant), lower the
    threshold; otherwise use plain Otsu.

    Args:
        gray: Grayscale image.

    Returns:
        Binary image.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = gray.size
    bright_ratio = hist[200:].sum() / total
    dark_ratio = hist[:50].sum() / total

    binary = otsu_binarization(gray)

    if bright_ratio > 0.7 or dark_ratio > 0.7:
        logger.debug(
            "Uneven lighting detected (bright=%.2f dark=%.2f), using adaptive",
            bright_ratio,
            dark_ratio,
        )
        binary = adaptive_binarization(gray)

    return binary


def contrast_optimize(gray: np.ndarray) -> np.ndarray:
    """Enhance contrast via CLAHE before binarization.

    Args:
        gray: Grayscale image.

    Returns:
        Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)