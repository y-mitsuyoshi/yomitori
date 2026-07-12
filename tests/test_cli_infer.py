"""Tests for CLI run_infer with mocked components."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np


def test_cli_run_infer_with_document_type(tmp_path):
    """run_infer should process an image with explicit document_type."""
    import cv2

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img_path = str(tmp_path / "test.png")
    cv2.imwrite(img_path, img)

    with patch("src.detection.doctr_detector.DoctrDetector") as mock_det_cls, \
         patch("src.recognition.trocr_recognizer.TrocrRecognizer") as mock_rec_cls, \
         patch("src.pipeline.ocr_engine.HomographyCorrector") as mock_hc:

        mock_det = MagicMock()
        mock_det.detect.return_value = []
        mock_det_cls.return_value = mock_det

        mock_rec = MagicMock()
        mock_rec.recognize_batch.return_value = []
        mock_rec_cls.return_value = mock_rec

        mock_hc.return_value.correct.return_value = (
            np.zeros((1512, 2400, 3), dtype=np.uint8),
            {"homography_applied": True, "corrected_image_size": [2400, 1512], "fallback_used": False},
        )

        from src.cli import run_infer
        import argparse

        args = argparse.Namespace(
            image=img_path,
            document_type="driver_license_front",
            device="cpu",
            model_path=None,
        )
        ret = run_infer(args)
        assert ret == 0


def test_cli_run_infer_auto_detect(tmp_path):
    """run_infer should auto-detect document type when not specified."""
    import cv2

    img = np.ones((1515, 2400, 3), dtype=np.uint8)
    img_path = str(tmp_path / "test.png")
    cv2.imwrite(img_path, img)

    with patch("src.detection.doctr_detector.DoctrDetector") as mock_det_cls, \
         patch("src.recognition.trocr_recognizer.TrocrRecognizer") as mock_rec_cls, \
         patch("src.pipeline.ocr_engine.HomographyCorrector") as mock_hc:

        mock_det = MagicMock()
        mock_det.detect.return_value = []
        mock_det_cls.return_value = mock_det

        mock_rec = MagicMock()
        mock_rec.recognize_batch.return_value = []
        mock_rec_cls.return_value = mock_rec

        mock_hc.return_value.correct.return_value = (
            np.zeros((1512, 2400, 3), dtype=np.uint8),
            {"homography_applied": True, "corrected_image_size": [2400, 1512], "fallback_used": False},
        )

        from src.cli import run_infer
        import argparse

        args = argparse.Namespace(
            image=img_path,
            document_type=None,
            device=None,
            model_path=None,
        )
        ret = run_infer(args)
        assert ret == 0


def test_cli_main_with_infer(tmp_path):
    """main() should dispatch to run_infer."""
    import cv2

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img_path = str(tmp_path / "test.png")
    cv2.imwrite(img_path, img)

    with patch("src.detection.doctr_detector.DoctrDetector") as mock_det_cls, \
         patch("src.recognition.trocr_recognizer.TrocrRecognizer") as mock_rec_cls, \
         patch("src.pipeline.ocr_engine.HomographyCorrector") as mock_hc:

        mock_det = MagicMock()
        mock_det.detect.return_value = []
        mock_det_cls.return_value = mock_det

        mock_rec = MagicMock()
        mock_rec.recognize_batch.return_value = []
        mock_rec_cls.return_value = mock_rec

        mock_hc.return_value.correct.return_value = (
            np.zeros((1512, 2400, 3), dtype=np.uint8),
            {"homography_applied": True, "corrected_image_size": [2400, 1512], "fallback_used": False},
        )

        old_argv = sys.argv
        sys.argv = ["yomitori", "infer", "--image", img_path, "--device", "cpu"]
        try:
            from src.cli import main
            ret = main()
            assert ret == 0
        finally:
            sys.argv = old_argv