"""Base classes for the document-type plugin system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Zone:
    """Field zone definition using normalized coordinates (0.0–1.0).

    Attributes:
        field_name: Machine-readable field identifier.
        label: Label text printed on the document (e.g. ``"氏名"``).
        x0: Left boundary (normalized).
        y0: Top boundary (normalized).
        x1: Right boundary (normalized).
        y1: Bottom boundary (normalized).
        label_remove: Remove the label string from the recognized text.
        date_format: Date format hint (e.g. ``"japanese_era"``).
        whitelist: Optional character whitelist string.
        normalize: Normalization hint (e.g. ``"fullwidth_to_halfwidth"``).
    """

    field_name: str
    label: str
    x0: float
    y0: float
    x1: float
    y1: float
    label_remove: bool = False
    date_format: str | None = None
    whitelist: str | None = None
    normalize: str | None = None

    def contains(self, cx: float, cy: float) -> bool:
        """Check whether a point is inside the zone.

        Args:
            cx: Normalized center-x.
            cy: Normalized center-y.

        Returns:
            True if (cx, cy) lies within [x0,x1]×[y0,y1].
        """
        return self.x0 <= cx <= self.x1 and self.y0 <= cy <= self.y1

    def distance_to(self, cx: float, cy: float) -> float:
        """Euclidean distance from (cx, cy) to the zone boundary.

        Args:
            cx: Normalized center-x.
            cy: Normalized center-y.

        Returns:
            Distance; 0.0 if inside.
        """
        dx = max(0.0, self.x0 - cx, cx - self.x1)
        dy = max(0.0, self.y0 - cy, cy - self.y1)
        return (dx * dx + dy * dy) ** 0.5


@dataclass
class ValidationRule:
    """Per-field validation rule.

    Attributes:
        pattern: Optional regex pattern that the field value must match.
        check_digit: Whether to run a check-digit verification.
        description: Human-readable description (used in warnings).
        required: Whether the field is required.
    """

    pattern: str | None = None
    check_digit: bool = False
    description: str = ""
    required: bool = True


class DocumentType(ABC):
    """Abstract base class for all document types.

    New document types are implemented by subclassing this and registering
    with ``DocumentTypeRegistry``. The core pipeline never needs modification
    when adding a new document type.
    """

    @property
    @abstractmethod
    def document_type_id(self) -> str:
        """Document type ID (e.g. ``"driver_license_front"``)."""
        ...

    @property
    @abstractmethod
    def card_aspect_ratio(self) -> float:
        """Card aspect ratio (width/height). Used for detection."""
        ...

    @property
    @abstractmethod
    def normalized_size(self) -> tuple[int, int]:
        """Normalized (width, height) after homography correction."""
        ...

    @property
    @abstractmethod
    def zones(self) -> dict[str, Zone]:
        """Field zone definitions keyed by field name."""
        ...

    @property
    @abstractmethod
    def validation_rules(self) -> dict[str, ValidationRule]:
        """Per-field validation rules keyed by field name."""
        ...

    @abstractmethod
    def detect(self, image: np.ndarray) -> float:
        """Score how well an image matches this document type.

        Args:
            image: Input image (BGR ndarray).

        Returns:
            Score in [0.0, 1.0]; higher means more likely.
        """
        ...

    @staticmethod
    def _check_aspect_ratio(
        image: np.ndarray, expected: float, tolerance: float = 0.05
    ) -> float:
        """Compute aspect-ratio similarity score.

        Args:
            image: Input image.
            expected: Expected width/height ratio.
            tolerance: Tolerance band; within it returns 1.0.

        Returns:
            Score in [0.0, 1.0].
        """
        h, w = image.shape[:2]
        actual = w / h
        diff = abs(actual - expected)
        if diff <= tolerance:
            return 1.0
        return max(0.0, 1.0 - diff / expected)