"""Tests for the OCREngine with mocked components."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.detection.doctr_detector import DetectionResult
from src.document_types.driver_license import DriverLicenseFront
from src.pipeline.field_extractor import FieldExtractor
from src.pipeline.ocr_engine import OCREngine
from src.postprocessing.validator import Validator


def _make_detection(cx: float, cy: float, text: str = "test") -> DetectionResult:
    """Create a mock DetectionResult."""
    img = np.zeros((20, 100, 3), dtype=np.uint8)
    return DetectionResult(
        bbox=[[0, 0], [1, 0], [1, 1], [0, 1]],
        bbox_pixels=[[0, 0], [100, 0], [100, 20], [0, 20]],
        line_image=img,
        center_normalized=(cx, cy),
        confidence=0.9,
    )


def _make_mock_detector(detections):
    """Create a mock detector that returns the given detections."""
    detector = MagicMock()
    detector.detect.return_value = detections
    return detector


def _make_mock_recognizer(results):
    """Create a mock recognizer that returns the given (text, conf) tuples."""
    recognizer = MagicMock()
    recognizer.recognize_batch.return_value = results
    return recognizer


def _make_engine(detector, recognizer, doc_type=None, confidence_threshold=0.7):
    """Create an OCREngine with mocked components."""
    if doc_type is None:
        doc_type = DriverLicenseFront()
    extractor = FieldExtractor(doc_type)
    validator = Validator()
    return OCREngine(
        doc_type=doc_type,
        detector=detector,
        recognizer=recognizer,
        extractor=extractor,
        validator=validator,
        confidence_threshold=confidence_threshold,
    )


@patch("src.pipeline.ocr_engine.HomographyCorrector")
def test_ocr_engine_success(mock_corrector_cls):
    """Full pipeline should return success for high-confidence fields."""
    mock_corrector_cls.return_value.correct.return_value = (
        np.zeros((1512, 2400, 3), dtype=np.uint8),
        {"homography_applied": True, "corrected_image_size": [2400, 1512], "fallback_used": False},
    )
    detections = [
        _make_detection(0.10, 0.05),
        _make_detection(0.55, 0.65),
    ]
    detector = _make_mock_detector(detections)
    recognizer = _make_mock_recognizer([
        ("氏名 山田太郎", 0.95),
        ("免許証番号 010203040506", 0.92),
    ])
    engine = _make_engine(detector, recognizer)
    result = engine.process(np.zeros((100, 200, 3), dtype=np.uint8))

    assert result["status"] in ("success", "partial")
    assert result["document_type"] == "driver_license_front"
    assert "overall_confidence" in result
    assert result["overall_confidence"] > 0
    assert "name" in result["fields"]
    assert "license_number" in result["fields"]


@patch("src.pipeline.ocr_engine.HomographyCorrector")
def test_ocr_engine_no_detections(mock_corrector_cls):
    """Pipeline should return failed when no text is detected."""
    mock_corrector_cls.return_value.correct.return_value = (
        np.zeros((1512, 2400, 3), dtype=np.uint8),
        {"homography_applied": True, "corrected_image_size": [2400, 1512], "fallback_used": False},
    )
    detector = _make_mock_detector([])
    recognizer = _make_mock_recognizer([])
    engine = _make_engine(detector, recognizer)
    result = engine.process(np.zeros((100, 200, 3), dtype=np.uint8))

    assert result["status"] == "failed"
    assert result["fields"] == {}
    assert result["overall_confidence"] == 0.0


@patch("src.pipeline.ocr_engine.HomographyCorrector")
def test_ocr_engine_low_confidence(mock_corrector_cls):
    """Pipeline should return partial when confidence is low."""
    mock_corrector_cls.return_value.correct.return_value = (
        np.zeros((1512, 2400, 3), dtype=np.uint8),
        {"homography_applied": True, "corrected_image_size": [2400, 1512], "fallback_used": False},
    )
    detections = [_make_detection(0.10, 0.05)]
    detector = _make_mock_detector(detections)
    recognizer = _make_mock_recognizer([("氏名 山田太郎", 0.3)])
    engine = _make_engine(detector, recognizer, confidence_threshold=0.7)
    result = engine.process(np.zeros((100, 200, 3), dtype=np.uint8))

    assert result["status"] == "partial"
    assert result["fields"]["name"]["low_confidence"] is True


@patch("src.pipeline.ocr_engine.HomographyCorrector")
def test_ocr_engine_fallback_failed(mock_corrector_cls):
    """Pipeline should return failed when fallback used and no fields."""
    mock_corrector_cls.return_value.correct.return_value = (
        np.zeros((1512, 2400, 3), dtype=np.uint8),
        {"homography_applied": False, "corrected_image_size": [2400, 1512], "fallback_used": True},
    )
    detector = _make_mock_detector([])
    recognizer = _make_mock_recognizer([])
    engine = _make_engine(detector, recognizer)
    result = engine.process(np.zeros((100, 200, 3), dtype=np.uint8))

    assert result["status"] == "failed"


@patch("src.pipeline.ocr_engine.HomographyCorrector")
def test_ocr_engine_multi_line_merge(mock_corrector_cls):
    """Two detections in the same zone should merge text."""
    mock_corrector_cls.return_value.correct.return_value = (
        np.zeros((1512, 2400, 3), dtype=np.uint8),
        {"homography_applied": True, "corrected_image_size": [2400, 1512], "fallback_used": False},
    )
    detections = [
        _make_detection(0.10, 0.05),
        _make_detection(0.15, 0.05),
    ]
    detector = _make_mock_detector(detections)
    recognizer = _make_mock_recognizer([
        ("氏名 山田", 0.9),
        ("太郎", 0.85),
    ])
    engine = _make_engine(detector, recognizer)
    result = engine.process(np.zeros((100, 200, 3), dtype=np.uint8))

    assert "name" in result["fields"]
    assert "山田" in result["fields"]["name"]["value"]
    assert "太郎" in result["fields"]["name"]["value"]
    # Confidence should be min of the two
    assert result["fields"]["name"]["confidence"] == 0.85


@patch("src.pipeline.ocr_engine.HomographyCorrector")
def test_ocr_engine_validation_failure(mock_corrector_cls):
    """Pipeline should return partial when validation fails."""
    mock_corrector_cls.return_value.correct.return_value = (
        np.zeros((1512, 2400, 3), dtype=np.uint8),
        {"homography_applied": True, "corrected_image_size": [2400, 1512], "fallback_used": False},
    )
    detections = [_make_detection(0.55, 0.65)]
    detector = _make_mock_detector(detections)
    recognizer = _make_mock_recognizer([("免許証番号 abc", 0.95)])
    engine = _make_engine(detector, recognizer)
    result = engine.process(np.zeros((100, 200, 3), dtype=np.uint8))

    assert result["status"] == "partial"
    assert result["fields"]["license_number"]["validation_passed"] is False
    assert any("license_number" in w for w in result["warnings"])


@patch("src.pipeline.ocr_engine.HomographyCorrector")
def test_ocr_engine_date_conversion(mock_corrector_cls):
    """Pipeline should convert Japanese era dates to ISO format."""
    mock_corrector_cls.return_value.correct.return_value = (
        np.zeros((1512, 2400, 3), dtype=np.uint8),
        {"homography_applied": True, "corrected_image_size": [2400, 1512], "fallback_used": False},
    )
    detections = [_make_detection(0.50, 0.05)]
    detector = _make_mock_detector(detections)
    recognizer = _make_mock_recognizer([("生年月日 昭和61年5月1日生", 0.95)])
    engine = _make_engine(detector, recognizer)
    result = engine.process(np.zeros((100, 200, 3), dtype=np.uint8))

    if "birth_date" in result["fields"]:
        field = result["fields"]["birth_date"]
        assert "raw" in field
        assert "iso" in field
        assert field["iso"] == "1986-05-01"


@patch("src.pipeline.ocr_engine.HomographyCorrector")
def test_ocr_engine_cross_field_validation(mock_corrector_cls):
    """Cross-field validation should detect inverted birth/expiry dates."""
    mock_corrector_cls.return_value.correct.return_value = (
        np.zeros((1512, 2400, 3), dtype=np.uint8),
        {"homography_applied": True, "corrected_image_size": [2400, 1512], "fallback_used": False},
    )
    # birth_date zone: x[0.45,0.85], y[0.00,0.10] → center (0.60, 0.05) (outside name zone x[0.00,0.55])
    # expiry_date zone: x[0.00,0.70], y[0.39,0.46] → center (0.10, 0.42)
    detections = [
        _make_detection(0.60, 0.05),
        _make_detection(0.10, 0.42),
    ]
    detector = _make_mock_detector(detections)
    recognizer = _make_mock_recognizer([
        ("生年月日 令和5年1月1日", 0.95),
        ("有効期限 1985年1月1日", 0.95),
    ])
    engine = _make_engine(detector, recognizer)
    result = engine.process(np.zeros((100, 200, 3), dtype=np.uint8))

    # birth_date ISO should be 2023-01-01, expiry_date value is "1985年1月1日" (西暦)
    # So birth (2023) > expiry (1985) → cross-field warning
    birth = result["fields"].get("birth_date", {})
    expiry = result["fields"].get("expiry_date", {})
    assert birth.get("iso") == "2023-01-01"
    assert "1985" in expiry.get("value", "")
    cross_warnings = [w for w in result["warnings"] if "生年月日が有効期限より後" in w]
    assert len(cross_warnings) == 1


@patch("src.pipeline.ocr_engine.HomographyCorrector")
def test_ocr_engine_unassigned_detection(mock_corrector_cls):
    """Detections that don't match any zone should be skipped."""
    mock_corrector_cls.return_value.correct.return_value = (
        np.zeros((1512, 2400, 3), dtype=np.uint8),
        {"homography_applied": True, "corrected_image_size": [2400, 1512], "fallback_used": False},
    )
    # Point at (0.99, 0.99) — far from all zones
    detections = [_make_detection(0.99, 0.99)]
    detector = _make_mock_detector(detections)
    recognizer = _make_mock_recognizer([("random text", 0.9)])
    engine = _make_engine(detector, recognizer, confidence_threshold=0.5)
    result = engine.process(np.zeros((100, 200, 3), dtype=np.uint8))

    # Field should not be assigned
    assert result["status"] == "failed" or len(result["fields"]) == 0


@patch("src.pipeline.ocr_engine.HomographyCorrector")
def test_ocr_engine_address_normalization(mock_corrector_cls):
    """Address field should have fullwidth_to_halfwidth normalization applied."""
    mock_corrector_cls.return_value.correct.return_value = (
        np.zeros((1512, 2400, 3), dtype=np.uint8),
        {"homography_applied": True, "corrected_image_size": [2400, 1512], "fallback_used": False},
    )
    # address zone: x[0.00,0.70], y[0.12,0.26] → center (0.10, 0.18)
    detections = [_make_detection(0.10, 0.18)]
    detector = _make_mock_detector(detections)
    recognizer = _make_mock_recognizer([("住所 東京都千代田区１－１", 0.95)])
    engine = _make_engine(detector, recognizer)
    result = engine.process(np.zeros((100, 200, 3), dtype=np.uint8))

    if "address" in result["fields"]:
        # Fullwidth digits should be converted to halfwidth
        assert "1" in result["fields"]["address"]["value"]