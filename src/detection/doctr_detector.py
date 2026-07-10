"""docTR-based text-line detection wrapper."""

from dataclasses import dataclass

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    """Single detected text line.

    Attributes:
        bbox: 4-corner polygon in normalized (0.0–1.0) coordinates,
            shape (4, 2), ordered (tl, tr, br, bl).
        bbox_pixels: 4-corner polygon in pixel coordinates (4, 2).
        line_image: Cropped line image (BGR ndarray).
        center_normalized: (cx, cy) in normalized coords.
        confidence: Detection confidence score.
    """

    bbox: list[list[float]]
    bbox_pixels: list[list[int]]
    line_image: np.ndarray
    center_normalized: tuple[float, float]
    confidence: float


class DoctrDetector:
    """docTR text-line detector.

    Uses only the detection stage; recognition is delegated to TrOCR.

    Args:
        det_arch: Detection architecture name.
        assume_straight_pages: Whether pages are assumed straight.
        device: ``"cuda"`` or ``"cpu"``.
        min_score: Minimum detection score to keep.
    """

    def __init__(
        self,
        det_arch: str = "db_resnet50",
        assume_straight_pages: bool = True,
        device: str = "cuda",
        min_score: float = 0.1,
    ) -> None:
        from doctr.models import ocr_predictor

        self.device = device
        self.min_score = min_score
        self.predictor = ocr_predictor(
            det_arch=det_arch,
            pretrained=True,
            assume_straight_pages=assume_straight_pages,
        ).to(device)
        logger.info("DoctrDetector initialized (det_arch=%s, device=%s)", det_arch, device)

    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        """Detect text lines and return cropped line images + coordinates.

        Args:
            image: Input BGR ndarray.

        Returns:
            List of DetectionResult, one per detected line.
        """
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        result = self.predictor([rgb])
        page = result.pages[0]

        detections: list[DetectionResult] = []
        for block in page.blocks:
            for line in block.lines:
                # line.geometry is ((x0,y0),(x1,y1)) normalized
                (x0, y0), (x1, y1) = line.geometry
                cx_n = (x0 + x1) / 2.0
                cy_n = (y0 + y1) / 2.0

                # Convert to pixel coords
                px0 = int(x0 * w)
                py0 = int(y0 * h)
                px1 = int(x1 * w)
                py1 = int(y1 * h)

                px0 = max(0, px0)
                py0 = max(0, py0)
                px1 = min(w, px1)
                py1 = min(h, py1)

                if px1 <= px0 or py1 <= py0:
                    continue

                line_img = image[py0:py1, px0:px1].copy()

                # 4-corner polygon (normalized + pixel)
                bbox_n = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                bbox_p = [[px0, py0], [px1, py0], [px1, py1], [px0, py1]]

                conf = float(line.confidence) if hasattr(line, "confidence") else 1.0
                if conf < self.min_score:
                    continue

                detections.append(
                    DetectionResult(
                        bbox=bbox_n,
                        bbox_pixels=bbox_p,
                        line_image=line_img,
                        center_normalized=(cx_n, cy_n),
                        confidence=conf,
                    )
                )

        logger.info("Detected %d text lines", len(detections))
        return detections