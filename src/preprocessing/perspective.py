"""4-point peak detection and perspective (homography) correction."""

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerspectiveResult:
    """Result of perspective correction.

    Attributes:
        corrected: Corrected image.
        src_points: Source quadrilateral (4, 2) array.
        homography_applied: Whether correction succeeded.
        fallback_used: Whether a fallback (resize) was used.
    """

    corrected: np.ndarray
    src_points: np.ndarray | None
    homography_applied: bool
    fallback_used: bool


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points into a consistent (tl, bl, br, tr) layout.

    The output order is: top-left, bottom-left, bottom-right, top-right.

    Args:
        pts: Array of shape (4, 2).

    Returns:
        Ordered array of shape (4, 2).
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # bottom-left
    rect[3] = pts[np.argmax(diff)]  # top-right
    return rect


def four_point_perspective(
    image: np.ndarray,
    src_points: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Apply perspective transform given 4 source points.

    Args:
        image: Input image.
        src_points: Source quadrilateral (4, 2).
        target_size: (width, height).

    Returns:
        Warped image.
    """
    w, h = target_size
    dst = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
    ordered_src = order_points(np.array(src_points, dtype=np.float32))
    M = cv2.getPerspectiveTransform(ordered_src, dst)
    return cv2.warpPerspective(image, M, (w, h))