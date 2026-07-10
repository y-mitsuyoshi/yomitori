"""HTTP inference server for Yomitori OCR engine.

SageMaker-like serving: start the server with `docker compose up serve`,
then send POST requests with base64-encoded images.

Endpoints:
  GET  /ping            — health check
  POST /invocations      — run OCR on an image (JSON with base64 image)
"""

import base64
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.utils.logger import get_logger

logger = get_logger(__name__)

_ENGINE = None
_MODEL = None


class InvocationRequest(BaseModel):
    """Request body for /invocations endpoint.

    Attributes:
        image: Base64-encoded image data (with or without data URI prefix).
        document_type: Optional document type ID. Auto-detected if omitted.
    """

    image: str
    document_type: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the OCR engine components at startup."""
    global _ENGINE, _MODEL
    import torch

    from src.detection.doctr_detector import DoctrDetector
    from src.document_types.driver_license import DriverLicenseFront
    from src.document_types.registry import DocumentTypeRegistry
    from src.pipeline.ocr_engine import OCREngine
    from src.postprocessing.validator import Validator
    from src.recognition.trocr_recognizer import TrocrRecognizer

    device = os.environ.get("YOMITORI_DEVICE", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    model_dir = os.environ.get("YOMITORI_MODEL_DIR", "/opt/ml/model")

    finetuned_path = None
    model_path = Path(model_dir)
    if model_path.exists():
        if (model_path / "config.json").exists():
            finetuned_path = str(model_path)
        elif (model_path / "trocr" / "config.json").exists():
            finetuned_path = str(model_path / "trocr")

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
    yield


app = FastAPI(title="Yomitori OCR Engine", version="1.0.0", lifespan=lifespan)


@app.get("/ping")
def ping() -> dict:
    """Health check endpoint.

    Returns:
        Status dict indicating the server is ready.
    """
    return {"status": "ok", "model_loaded": _MODEL is not None}


@app.post("/invocations")
async def invocations(req: InvocationRequest) -> dict:
    """Run OCR on a base64-encoded image.

    Args:
        req: Request body with base64 image and optional document type.

    Returns:
        OCR result as JSON.

    Raises:
        HTTPException: If image decoding or inference fails.
    """
    import cv2
    import numpy as np

    from src.pipeline.field_extractor import FieldExtractor
    from src.pipeline.ocr_engine import OCREngine
    from src.utils.image_utils import decode_base64_image

    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image = decode_base64_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    registry = _MODEL["registry"]
    if req.document_type:
        doc_type = registry.get_by_id(req.document_type)
    else:
        doc_type = registry.detect(image)

    extractor = FieldExtractor(doc_type=doc_type)

    engine = OCREngine(
        doc_type=doc_type,
        detector=_MODEL["detector"],
        recognizer=_MODEL["recognizer"],
        extractor=extractor,
        validator=_MODEL["validator"],
    )

    result = engine.process(image)
    return result


def main() -> None:
    """Start the inference server."""
    uvicorn.run(
        "sagemaker.serve:app",
        host="0.0.0.0",
        port=int(os.environ.get("YOMITORI_PORT", "8080")),
        log_level="info",
    )


if __name__ == "__main__":
    main()