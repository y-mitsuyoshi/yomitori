"""SageMaker inference entry point.

Implements the four SageMaker serving functions: model_fn, input_fn,
predict_fn, output_fn.
"""

import base64
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

_MODEL: dict | None = None


def model_fn(model_dir: str) -> dict:
    """Load the model artifacts at SageMaker endpoint startup.

    Args:
        model_dir: Path to the model directory (/opt/ml/model).

    Returns:
        Dict containing the OCR engine components.
    """
    global _MODEL
    from src.detection.doctr_detector import DoctrDetector
    from src.document_types.driver_license import DriverLicenseFront
    from src.document_types.registry import DocumentTypeRegistry
    from src.pipeline.field_extractor import FieldExtractor
    from src.pipeline.ocr_engine import OCREngine
    from src.postprocessing.validator import Validator
    from src.recognition.trocr_recognizer import TrocrRecognizer

    device = os.environ.get("SAGEMAKER_DEVICE", "cuda")
    finetuned_path = str(Path(model_dir) / "trocr") if (Path(model_dir) / "trocr").exists() else None
    if finetuned_path is None and Path(model_dir).exists():
        # Check if model files are directly in model_dir
        if (Path(model_dir) / "config.json").exists():
            finetuned_path = model_dir

    registry = DocumentTypeRegistry()
    registry.register(DriverLicenseFront())

    detector = DoctrDetector(device=device)
    recognizer = TrocrRecognizer(device=device, finetuned_path=finetuned_path)

    _MODEL = {
        "registry": registry,
        "detector": detector,
        "recognizer": recognizer,
        "validator": Validator(),
    }
    logger.info("Model loaded (finetuned_path=%s, device=%s)", finetuned_path, device)
    return _MODEL


def input_fn(request_body: bytes, content_type: str) -> np.ndarray:
    """Deserialize the incoming request.

    Args:
        request_body: Raw request bytes.
        content_type: MIME type.

    Returns:
        Decoded image as BGR ndarray.

    Raises:
        ValueError: For unsupported content types.
    """
    from src.utils.image_utils import decode_base64_image, decode_image

    if content_type == "image/jpeg" or content_type == "image/png":
        return decode_image(request_body)
    if content_type == "application/json":
        data = json.loads(request_body)
        if "image" in data:
            return decode_base64_image(data["image"])
        raise ValueError("JSON must contain 'image' (base64)")
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: np.ndarray, model: dict) -> dict:
    """Run the OCR pipeline on the input image.

    Args:
        input_data: Input image (BGR ndarray).
        model: Model dict from model_fn.

    Returns:
        OCR result dict.
    """
    from src.pipeline.field_extractor import FieldExtractor
    from src.pipeline.ocr_engine import OCREngine

    registry = model["registry"]
    doc_type = registry.detect(input_data)
    extractor = FieldExtractor(doc_type=doc_type)

    engine = OCREngine(
        doc_type=doc_type,
        detector=model["detector"],
        recognizer=model["recognizer"],
        extractor=extractor,
        validator=model["validator"],
    )
    return engine.process(input_data)


def output_fn(prediction: dict, accept: str) -> str | bytes:
    """Serialize the prediction for the response.

    Args:
        prediction: Result dict.
        accept: Requested MIME type.

    Returns:
        Serialized response.
    """
    if accept in ("application/json", "text/html", "*/*", ""):
        return json.dumps(prediction, ensure_ascii=False)
    raise ValueError(f"Unsupported accept type: {accept}")