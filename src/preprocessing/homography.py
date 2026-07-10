"""Homography (perspective) correction for skewed/trapezoid card images."""

from dataclasses import dataclass

import cv2
import numpy as np

from src.preprocessing.binarization import dynamic_binarization, contrast_optimize
from src.preprocessing.perspective import four_point_perspective, order_points
from src.utils.image_utils import resize_image
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HomographyInfo:
    """Metadata about the correction result.

    Attributes:
        homography_applied: Whether correction was applied.
        fallback_used: Whether a fallback (resize) was used.
        corrected_size: (width, height) of the corrected image.
    """

    homography_applied: bool
    fallback_used: bool
    corrected_size: tuple[int, int]

    def to_dict(self) -> dict:
        return {
            "homography_applied": self.homography_applied,
            "corrected_image_size": list(self.corrected_size),
            "fallback_used": self.fallback_used,
        }


class HomographyCorrector:
    """Detect a card region and apply perspective correction.

    Args:
        target_size: Target (width, height) after correction.
        min_contour_area_ratio: Minimum contour area vs. image area.
        epsilon_factor: Initial epsilon factor for approxPolyDP.
        epsilon_min: Min epsilon factor for search loop.
        epsilon_max: Max epsilon factor for search loop.
        epsilon_step: Step for epsilon search.
    """

    def __init__(
        self,
        target_size: tuple[int, int],
        min_contour_area_ratio: float = 0.30,
        epsilon_factor: float = 0.02,
        epsilon_min: float = 0.01,
        epsilon_max: float = 0.15,
        epsilon_step: float = 0.005,
    ) -> None:
        self.target_size = target_size
        self.min_contour_area_ratio = min_contour_area_ratio
        self.epsilon_factor = epsilon_factor
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_step = epsilon_step

    def correct(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """Apply homography correction to an image.

        Args:
            image: Input BGR image.

        Returns:
            Tuple of (corrected_image, info_dict).
        """
        h, w = image.shape[:2]
        img_area = h * w

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = contrast_optimize(gray)
        binary = dynamic_binarization(enhanced)

        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning("No contours found; falling back to resize")
            return self._fallback(image)

        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) < img_area * self.min_contour_area_ratio:
            logger.warning(
                "Largest contour too small (%.1f%%); falling back",
                100 * cv2.contourArea(max_contour) / img_area,
            )
            return self._fallback(image)

        arc = cv2.arcLength(max_contour, True)
        approx = self._find_four_points(max_contour, arc)
        if approx is None:
            logger.warning("4-point approximation failed; falling back")
            return self._fallback(image)

        ordered = order_points(approx)
        corrected = four_point_perspective(image, ordered, self.target_size)
        info = HomographyInfo(
            homography_applied=True,
            fallback_used=False,
            corrected_size=self.target_size,
        )
        logger.info("Homography correction applied successfully")
        return corrected, info.to_dict()

    def _find_four_points(self, contour: np.ndarray, arc: float) -> np.ndarray | None:
        """Search for a 4-point approximation of the contour.

        Args:
            contour: OpenCV contour.
            arc: Arc length of the contour.

        Returns:
            Array of 4 points (4, 2) or None.
        """
        approx = cv2.approxPolyDP(contour, self.epsilon_factor * arc, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)

        for factor in np.arange(
            self.epsilon_min, self.epsilon_max, self.epsilon_step
        ):
            eps = factor * arc
            approx = cv2.approxPolyDP(contour, eps, True)
            if len(approx) == 4:
                logger.debug("4-point approximation at epsilon=%.4f", factor)
                return approx.reshape(4, 2)
        return None

    def _fallback(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """Resize original image to target size as a fallback.

        Args:
            image: Input image.

        Returns:
            Tuple of (resized_image, info_dict).
        """
        resized = resize_image(image, self.target_size)
        info = HomographyInfo(
            homography_applied=False,
            fallback_used=True,
            corrected_size=self.target_size,
        )
        logger.warning("Fallback: original image resized to %s", self.target_size)
        return resized, info.to_dict()

    def visualize(self, image: np.ndarray) -> None:
        """Visualize each preprocessing step (for Jupyter/debug).

        Args:
            image: Input BGR image.
        """
        from src.utils.image_utils import visualize_preprocessing

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = contrast_optimize(gray)
        binary = dynamic_binarization(enhanced)
        corrected, _ = self.correct(image)
        visualize_preprocessing(image, gray, binary, None, corrected)