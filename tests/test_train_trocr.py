"""Tests for TrOCR multilingual training builders."""

from unittest.mock import MagicMock, patch
import pytest

from training.train_trocr import (
    build_multilingual_processor,
    build_multilingual_model,
)


def test_build_multilingual_processor_bert_japanese():
    """build_multilingual_processor should correctly configure special tokens for BERT Japanese."""
    with patch("transformers.TrOCRProcessor.from_pretrained") as mock_proc, \
         patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:

        # Mock tokenizer which doesn't have bos_token or eos_token initially
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = None
        mock_tokenizer.bos_token_id = None
        mock_tokenizer.eos_token = None
        mock_tokenizer.eos_token_id = None
        mock_tokenizer.cls_token = "[CLS]"
        mock_tokenizer.cls_token_id = 2
        mock_tokenizer.sep_token = "[SEP]"
        mock_tokenizer.sep_token_id = 3
        mock_tok.return_value = mock_tokenizer

        mock_processor_inst = MagicMock()
        mock_proc.return_value = mock_processor_inst

        processor = build_multilingual_processor(
            base_model="dummy-base-model",
            decoder_tokenizer="dummy-tokenizer",
        )

        assert processor == mock_processor_inst
        # Verify that special tokens are mapped to cls_token and sep_token
        assert mock_tokenizer.bos_token == "[CLS]"
        assert mock_tokenizer.bos_token_id == 2
        assert mock_tokenizer.eos_token == "[SEP]"
        assert mock_tokenizer.eos_token_id == 3
        assert processor.tokenizer == mock_tokenizer


def test_build_multilingual_model_bert_japanese():
    """build_multilingual_model should configure vocab size, config, and generation_config correctly."""
    with patch("transformers.VisionEncoderDecoderModel.from_pretrained") as mock_model, \
         patch("transformers.TrOCRProcessor.from_pretrained") as mock_proc, \
         patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:

        # Mock tokenizer without bos/eos
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = None
        mock_tokenizer.bos_token_id = None
        mock_tokenizer.eos_token = None
        mock_tokenizer.eos_token_id = None
        mock_tokenizer.cls_token = "[CLS]"
        mock_tokenizer.cls_token_id = 2
        mock_tokenizer.sep_token = "[SEP]"
        mock_tokenizer.sep_token_id = 3
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.__len__.return_value = 32000
        mock_tok.return_value = mock_tokenizer

        # Mock model and configs
        mock_model_inst = MagicMock()
        mock_model_inst.config = MagicMock()
        mock_model_inst.config.decoder = MagicMock()
        mock_model_inst.generation_config = MagicMock()
        mock_model.return_value = mock_model_inst

        mock_processor_inst = MagicMock()
        mock_proc.return_value = mock_processor_inst

        model, processor = build_multilingual_model(
            base_model="dummy-base",
            decoder_tokenizer="dummy-tokenizer",
        )

        assert model == mock_model_inst
        assert processor == mock_processor_inst

        # Verify tokenizer mapping
        assert mock_tokenizer.bos_token == "[CLS]"
        assert mock_tokenizer.bos_token_id == 2
        assert mock_tokenizer.eos_token == "[SEP]"
        assert mock_tokenizer.eos_token_id == 3

        # Verify model config resize and token assignments
        assert mock_model_inst.config.decoder.vocab_size == 32000
        mock_model_inst.decoder.resize_token_embeddings.assert_called_once_with(32000)
        assert mock_model_inst.config.pad_token_id == 0
        assert mock_model_inst.config.decoder_start_token_id == 2
        assert mock_model_inst.config.eos_token_id == 3
        assert mock_model_inst.config.vocab_size == 32000

        # Verify generation_config update
        assert mock_model_inst.generation_config.pad_token_id == 0
        assert mock_model_inst.generation_config.decoder_start_token_id == 2
        assert mock_model_inst.generation_config.eos_token_id == 3


def test_build_multilingual_model_continuation(tmp_path):
    """build_multilingual_model should skip vocab resize for continuation training.

    When base_model is a local directory containing training_info.json,
    the function should use the saved tokenizer and NOT call resize_token_embeddings.
    """
    import json

    # Create a fake local model directory with training_info.json
    model_dir = tmp_path / "v2.1"
    model_dir.mkdir()
    (model_dir / "training_info.json").write_text(json.dumps({
        "base_model": "/opt/ml/model/v2",
        "decoder_tokenizer": "cl-tohoku/bert-base-japanese-v3",
    }))

    with patch("transformers.VisionEncoderDecoderModel.from_pretrained") as mock_model, \
         patch("transformers.TrOCRProcessor.from_pretrained") as mock_proc, \
         patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:

        # Mock tokenizer (already has bos/eos from previous training)
        mock_tokenizer = MagicMock()
        mock_tokenizer.bos_token = "[CLS]"
        mock_tokenizer.bos_token_id = 2
        mock_tokenizer.eos_token = "[SEP]"
        mock_tokenizer.eos_token_id = 3
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.__len__.return_value = 32768
        mock_tok.return_value = mock_tokenizer

        mock_model_inst = MagicMock()
        mock_model_inst.config = MagicMock()
        mock_model_inst.config.decoder = MagicMock()
        mock_model_inst.generation_config = MagicMock()
        mock_model.return_value = mock_model_inst

        mock_processor_inst = MagicMock()
        mock_proc.return_value = mock_processor_inst

        model, processor = build_multilingual_model(
            base_model=str(model_dir),
            decoder_tokenizer="cl-tohoku/bert-base-japanese-v3",
        )

        assert model == mock_model_inst
        assert processor == mock_processor_inst

        # Key assertion: resize_token_embeddings should NOT be called
        mock_model_inst.decoder.resize_token_embeddings.assert_not_called()

        # Tokenizer should be loaded from the local model dir, not the decoder_tokenizer arg
        mock_tok.assert_called_with(str(model_dir))
        assert processor.tokenizer == mock_tokenizer
