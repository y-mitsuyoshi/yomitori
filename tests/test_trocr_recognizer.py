"""Tests for TrocrRecognizer with mocked transformers."""

from unittest.mock import MagicMock, patch

import numpy as np


def test_trocr_recognizer_init_base():
    """TrocrRecognizer should load base model on CPU."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("transformers.TrOCRProcessor.from_pretrained") as mock_proc, \
         patch("transformers.VisionEncoderDecoderModel.from_pretrained") as mock_model:

        mock_proc.return_value = MagicMock()
        mock_model_inst = MagicMock()
        mock_model.return_value = mock_model_inst

        from src.recognition.trocr_recognizer import TrocrRecognizer

        recognizer = TrocrRecognizer(device="cpu", finetuned_path=None)
        assert recognizer.device == "cpu"
        assert recognizer.batch_size == 8
        mock_proc.assert_called_once()
        mock_model.assert_called_once()


def test_trocr_recognizer_init_finetuned():
    """TrocrRecognizer should load fine-tuned model from path."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("transformers.TrOCRProcessor.from_pretrained") as mock_proc, \
         patch("transformers.VisionEncoderDecoderModel.from_pretrained") as mock_model:

        mock_proc.return_value = MagicMock()
        mock_model_inst = MagicMock()
        mock_model.return_value = mock_model_inst

        from src.recognition.trocr_recognizer import TrocrRecognizer

        recognizer = TrocrRecognizer(device="cpu", finetuned_path="/some/path")
        assert recognizer.device == "cpu"
        # Should be called with the finetuned_path, not the base model name
        mock_proc.assert_called_with("/some/path")
        mock_model.assert_called_with("/some/path")


def test_trocr_recognizer_cuda_fallback():
    """TrocrRecognizer should fall back to CPU when CUDA unavailable."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("transformers.TrOCRProcessor.from_pretrained") as mock_proc, \
         patch("transformers.VisionEncoderDecoderModel.from_pretrained") as mock_model:

        mock_proc.return_value = MagicMock()
        mock_model_inst = MagicMock()
        mock_model.return_value = mock_model_inst

        from src.recognition.trocr_recognizer import TrocrRecognizer

        recognizer = TrocrRecognizer(device="cuda")
        assert recognizer.device == "cpu"


def test_trocr_recognizer_recognize():
    """recognize should return (text, confidence) tuple."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("transformers.TrOCRProcessor.from_pretrained") as mock_proc, \
         patch("transformers.VisionEncoderDecoderModel.from_pretrained") as mock_model:

        mock_processor = MagicMock()
        mock_proc.return_value = mock_processor

        mock_model_inst = MagicMock()
        mock_model.return_value = mock_model_inst

        from src.recognition.trocr_recognizer import TrocrRecognizer
        import torch

        recognizer = TrocrRecognizer(device="cpu")

        # Mock generate output
        mock_generated = MagicMock()
        mock_generated.sequences = torch.tensor([[1, 2, 3]])
        mock_scores = [torch.randn(1, 10)]
        mock_generated.scores = mock_scores
        mock_model_inst.generate.return_value = mock_generated

        # Mock batch_decode
        mock_processor.batch_decode.return_value = ["test text"]

        img = np.zeros((20, 100, 3), dtype=np.uint8)
        text, conf = recognizer.recognize(img)
        assert text == "test text"
        assert 0.0 <= conf <= 1.0


def test_trocr_recognizer_recognize_no_scores():
    """recognize should return confidence 0.0 when no scores."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("transformers.TrOCRProcessor.from_pretrained") as mock_proc, \
         patch("transformers.VisionEncoderDecoderModel.from_pretrained") as mock_model:

        mock_processor = MagicMock()
        mock_proc.return_value = mock_processor

        mock_model_inst = MagicMock()
        mock_model.return_value = mock_model_inst

        from src.recognition.trocr_recognizer import TrocrRecognizer
        import torch

        recognizer = TrocrRecognizer(device="cpu")

        mock_generated = MagicMock()
        mock_generated.sequences = torch.tensor([[1, 2]])
        mock_generated.scores = []
        mock_model_inst.generate.return_value = mock_generated

        mock_processor.batch_decode.return_value = ["test"]

        img = np.zeros((20, 100, 3), dtype=np.uint8)
        text, conf = recognizer.recognize(img)
        assert text == "test"
        assert conf == 0.0


def test_trocr_recognizer_recognize_batch():
    """recognize_batch should return list of (text, confidence) tuples."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("transformers.TrOCRProcessor.from_pretrained") as mock_proc, \
         patch("transformers.VisionEncoderDecoderModel.from_pretrained") as mock_model:

        mock_processor = MagicMock()
        mock_proc.return_value = mock_processor

        mock_model_inst = MagicMock()
        mock_model.return_value = mock_model_inst

        from src.recognition.trocr_recognizer import TrocrRecognizer
        import torch

        recognizer = TrocrRecognizer(device="cpu", batch_size=2)

        mock_generated = MagicMock()
        mock_generated.sequences = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mock_scores = [torch.randn(2, 10), torch.randn(2, 10)]
        mock_generated.scores = mock_scores
        mock_model_inst.generate.return_value = mock_generated

        mock_processor.batch_decode.return_value = ["text1", "text2"]

        images = [np.zeros((20, 100, 3), dtype=np.uint8) for _ in range(2)]
        results = recognizer.recognize_batch(images)

        assert len(results) == 2
        assert results[0][0] == "text1"
        assert results[1][0] == "text2"


def test_trocr_recognizer_recognize_batch_no_scores():
    """recognize_batch should return 0.0 confidence when no scores."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("transformers.TrOCRProcessor.from_pretrained") as mock_proc, \
         patch("transformers.VisionEncoderDecoderModel.from_pretrained") as mock_model:

        mock_processor = MagicMock()
        mock_proc.return_value = mock_processor

        mock_model_inst = MagicMock()
        mock_model.return_value = mock_model_inst

        from src.recognition.trocr_recognizer import TrocrRecognizer
        import torch

        recognizer = TrocrRecognizer(device="cpu", batch_size=2)

        mock_generated = MagicMock()
        mock_generated.sequences = torch.tensor([[1, 2], [4, 5]])
        mock_generated.scores = []
        mock_model_inst.generate.return_value = mock_generated

        mock_processor.batch_decode.return_value = ["text1", "text2"]

        images = [np.zeros((20, 100, 3), dtype=np.uint8) for _ in range(2)]
        results = recognizer.recognize_batch(images)

        assert len(results) == 2
        assert results[0][1] == 0.0
        assert results[1][1] == 0.0