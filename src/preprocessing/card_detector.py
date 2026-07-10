"""Generic card detector using contour detection and aspect-ratio matching."""

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_card_region(
    image: np.ndarray,
    min_area_ratio: float = 0.30,
    epsilon_factor: float = 0.02,
    epsilon_min: float = 0.01,
    epsilon_max: float = 0.15,
    epsilon_step: float = 0.005,
) -> np.ndarray | None:
    """Detect a card-like region and return its 4-corner approximation.

    Args:
        image: Input image (BGR).
        min_area_ratio: Minimum contour area ratio vs. image area.
        epsilon_factor: Initial epsilon factor for approxPolyDP.
        epsilon_min: Min epsilon factor to try.
        epsilon_max: Max epsilon factor to try.
        epsilon_step: Step for epsilon search.

    Returns:
        Array of 4 points (4, 2) or None if detection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("No contours found for card detection")
        return None

    img_area = image.shape[0] * image.shape[1]
    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) < img_area * min_area_ratio:
        logger.warning(
            "Largest contour area too small: %.1f%% of image",
            100 * cv2.contourArea(max_contour) / img_area,
        )
        return None

    arc = cv2.arcLength(max_contour, True)
    epsilon = epsilon_factor * arc
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    if len(approx) == 4:
        return approx.reshape(4, 2)

    for factor in np.arange(epsilon_min, epsilon_max, epsilon_step):
        eps = factor * arc
        approx = cv2.approxPolyDP(max_contour, eps, True)
        if len(approx) == 4:
            logger.debug("4-point approximation found at epsilon=%.4f", factor)
            return approx.reshape(4, 2)

    logger.warning("Could not approximate 4 corners")
    return None


def classify_by_aspect_ratio(
    image: np.ndarray,
    aspect_ratios: dict[str, float],
    tolerance: float = 0.05,
) -> str | None:
    """Classify a card region by its aspect ratio.

    Args:
        image: Input image.
        aspect_ratios: Mapping of doc-type-id → expected ratio.
        tolerance: Allowed deviation.

    Returns:
        Best-matching document type ID or None.
    """
    h, w = image.shape[:2]
    actual = w / h
    best_id: str | None = None
    best_diff = float("inf")
    for doc_id, expected in aspect_ratios.items():
        diff = abs(actual - expected)
        if diff <= tolerance and diff < best_diff:
            best_diff = diff
            best_id = doc_id
    return best_id