"""Zone-based field classification."""

import math

from src.document_types.base import DocumentType, Zone
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MARGIN = 0.04


class FieldExtractor:
    """Assign text boxes to document fields via zone definitions.

    Uses a two-pass strategy:
      1. Strict containment (center inside a zone).
      2. Nearest zone within a margin.

    Args:
        doc_type: The document type whose zones to use.
        margin: Max normalized distance for the second pass.
    """

    def __init__(self, doc_type: DocumentType, margin: float = _DEFAULT_MARGIN) -> None:
        self.zones: dict[str, Zone] = doc_type.zones
        self.margin = margin

    def assign(self, normalized_cx: float, normalized_cy: float) -> str | None:
        """Assign a normalized center point to a field name.

        Args:
            normalized_cx: Center x in [0, 1].
            normalized_cy: Center y in [0, 1].

        Returns:
            Field name or None if no zone matches.
        """
        # Pass 1: strict containment
        for field_name, zone in self.zones.items():
            if zone.contains(normalized_cx, normalized_cy):
                return field_name

        # Pass 2: nearest zone within margin
        best_zone: str | None = None
        best_dist = self.margin
        for field_name, zone in self.zones.items():
            dist = zone.distance_to(normalized_cx, normalized_cy)
            if dist < best_dist:
                best_dist = dist
                best_zone = field_name

        if best_zone is not None:
            logger.debug(
                "Pass-2 assignment: %s (dist=%.3f)", best_zone, best_dist
            )
        return best_zone