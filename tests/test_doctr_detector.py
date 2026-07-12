"""Tests for DoctrDetector with mocked docTR models."""

from unittest.mock import MagicMock, patch

import numpy as np


def test_doctr_detector_init_cpu():
    """DoctrDetector should initialize on CPU."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("doctr.models.ocr_predictor") as mock_predictor:
        mock_predictor.return_value.to.return_value = MagicMock()
        from src.detection.doctr_detector import DoctrDetector

        detector = DoctrDetector(device="cpu")
        assert detector.device == "cpu"
        assert detector.min_score == 0.1


def test_doctr_detector_cuda_fallback():
    """DoctrDetector should fall back to CPU when CUDA unavailable."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("doctr.models.ocr_predictor") as mock_predictor:
        mock_predictor.return_value.to.return_value = MagicMock()
        from src.detection.doctr_detector import DoctrDetector

        detector = DoctrDetector(device="cuda")
        assert detector.device == "cpu"


def test_doctr_detector_detect():
    """detect should return DetectionResult list."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("doctr.models.ocr_predictor") as mock_predictor:
        mock_predictor.return_value.to.return_value = MagicMock()
        from src.detection.doctr_detector import DoctrDetector

        detector = DoctrDetector(device="cpu")

        # Mock a page with one block containing one line
        mock_line = MagicMock()
        mock_line.geometry = ((0.1, 0.1), (0.5, 0.2))
        mock_line.confidence = 0.9

        mock_block = MagicMock()
        mock_block.lines = [mock_line]

        mock_page = MagicMock()
        mock_page.blocks = [mock_block]

        mock_result = MagicMock()
        mock_result.pages = [mock_page]

        detector.predictor = MagicMock(return_value=mock_result)

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        detections = detector.detect(img)

        assert len(detections) == 1
        cx, cy = detections[0].center_normalized
        assert abs(cx - 0.3) < 1e-6
        assert abs(cy - 0.15) < 1e-6
        assert detections[0].confidence == 0.9


def test_doctr_detector_detect_empty():
    """detect should return empty list when no blocks found."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("doctr.models.ocr_predictor") as mock_predictor:
        mock_predictor.return_value.to.return_value = MagicMock()
        from src.detection.doctr_detector import DoctrDetector

        detector = DoctrDetector(device="cpu")

        mock_page = MagicMock()
        mock_page.blocks = []

        mock_result = MagicMock()
        mock_result.pages = [mock_page]

        detector.predictor = MagicMock(return_value=mock_result)

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        detections = detector.detect(img)

        assert len(detections) == 0


def test_doctr_detector_low_score_filtered():
    """detect should filter out detections below min_score."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("doctr.models.ocr_predictor") as mock_predictor:
        mock_predictor.return_value.to.return_value = MagicMock()
        from src.detection.doctr_detector import DoctrDetector

        detector = DoctrDetector(device="cpu", min_score=0.5)

        mock_line = MagicMock()
        mock_line.geometry = ((0.1, 0.1), (0.5, 0.2))
        mock_line.confidence = 0.3

        mock_block = MagicMock()
        mock_block.lines = [mock_line]

        mock_page = MagicMock()
        mock_page.blocks = [mock_block]

        mock_result = MagicMock()
        mock_result.pages = [mock_page]

        detector.predictor = MagicMock(return_value=mock_result)

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        detections = detector.detect(img)

        assert len(detections) == 0


def test_doctr_detector_degenerate_bbox():
    """detect should skip degenerate bounding boxes."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("doctr.models.ocr_predictor") as mock_predictor:
        mock_predictor.return_value.to.return_value = MagicMock()
        from src.detection.doctr_detector import DoctrDetector

        detector = DoctrDetector(device="cpu")

        # Degenerate: zero-width box
        mock_line = MagicMock()
        mock_line.geometry = ((0.1, 0.1), (0.1, 0.2))
        mock_line.confidence = 0.9

        mock_block = MagicMock()
        mock_block.lines = [mock_line]

        mock_page = MagicMock()
        mock_page.blocks = [mock_block]

        mock_result = MagicMock()
        mock_result.pages = [mock_page]

        detector.predictor = MagicMock(return_value=mock_result)

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        detections = detector.detect(img)

        assert len(detections) == 0


def test_doctr_detector_no_confidence_attr():
    """detect should default confidence to 1.0 when attribute missing."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("doctr.models.ocr_predictor") as mock_predictor:
        mock_predictor.return_value.to.return_value = MagicMock()
        from src.detection.doctr_detector import DoctrDetector

        detector = DoctrDetector(device="cpu")

        mock_line = MagicMock()
        mock_line.geometry = ((0.1, 0.1), (0.5, 0.2))
        del mock_line.confidence  # Remove attribute to trigger hasattr=False

        mock_block = MagicMock()
        mock_block.lines = [mock_line]

        mock_page = MagicMock()
        mock_page.blocks = [mock_block]

        mock_result = MagicMock()
        mock_result.pages = [mock_page]

        detector.predictor = MagicMock(return_value=mock_result)

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        detections = detector.detect(img)

        assert len(detections) == 1
        assert detections[0].confidence == 1.0