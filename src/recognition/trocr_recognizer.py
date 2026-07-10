"""TrOCR-based text recognition wrapper."""

import logging
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrocrRecognizer:
    """TrOCR text recognizer.

    Args:
        model_name: HuggingFace model name for the base model.
        device: ``"cuda"`` or ``"cpu"``.
        finetuned_path: Optional path to a fine-tuned model directory.
        batch_size: Batch size for batch recognition.
    """

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-printed",
        device: str = "cuda",
        finetuned_path: Optional[str] = None,
        batch_size: int = 8,
    ) -> None:
        import torch
        from transformers import (
            TrOCRProcessor,
            VisionEncoderDecoderModel,
        )

        self.device = device
        self.batch_size = batch_size
        self.processor = TrOCRProcessor.from_pretrained(model_name)

        if finetuned_path:
            logger.info("Loading fine-tuned TrOCR from %s", finetuned_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(finetuned_path)
        else:
            logger.info("Loading base TrOCR model: %s", model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        self.model.to(device)
        self.model.eval()
        self._torch = torch
        logger.info("TrocrRecognizer ready (device=%s)", device)

    def recognize(self, line_image: np.ndarray) -> tuple[str, float]:
        """Recognize text from a single line image.

        Args:
            line_image: BGR ndarray of a single text line.

        Returns:
            Tuple of (recognized_text, confidence_score).
        """
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with self._torch.no_grad():
            generated = self.model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode text
        text = self.processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]

        # Estimate confidence from mean token score
        scores = generated.scores  # tuple of tensors
        if scores:
            probs = [self._torch.softmax(s, dim=-1) for s in scores]
            max_probs = [p.max().item() for p in probs]
            confidence = sum(max_probs) / len(max_probs) if max_probs else 0.0
        else:
            confidence = 0.0

        logger.debug("Recognized: %r (conf=%.3f)", text, confidence)
        return text, float(confidence)

    def recognize_batch(self, line_images: list[np.ndarray]) -> list[tuple[str, float]]:
        """Recognize multiple line images in batches for GPU efficiency.

        Args:
            line_images: List of BGR ndarrays.

        Returns:
            List of (text, confidence) tuples.
        """
        import cv2
        from PIL import Image

        results: list[tuple[str, float]] = []
        for start in range(0, len(line_images), self.batch_size):
            batch = line_images[start : start + self.batch_size]
            pil_imgs = [
                Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in batch
            ]
            pixel_values = self.processor(pil_imgs, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            with self._torch.no_grad():
                generated = self.model.generate(
                    pixel_values,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            texts = self.processor.batch_decode(
                generated.sequences, skip_special_tokens=True
            )

            # Compute per-sample confidence
            scores = generated.scores
            if scores:
                # scores[i] shape: (batch, vocab)
                seq_len = len(scores)
                for j in range(len(batch)):
                    max_probs = []
                    for i in range(seq_len):
                        if generated.sequences[j, i + 1] < scores[i].shape[1]:
                            p = self._torch.softmax(scores[i][j], dim=-1)
                            max_probs.append(p.max().item())
                    conf = sum(max_probs) / len(max_probs) if max_probs else 0.0
                    results.append((texts[j], float(conf)))
            else:
                for j, t in enumerate(texts):
                    results.append((t, 0.0))

        logger.info("Batch-recognized %d lines", len(results))
        return results