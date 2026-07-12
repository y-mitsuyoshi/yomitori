"""Command-line interface for the Yomitori OCR engine."""

import argparse
import json
import sys

from src.document_types.driver_license import DriverLicenseFront
from src.document_types.mynumber_card import MyNumberCardFront
from src.document_types.registry import DocumentTypeRegistry
from src.utils.image_utils import decode_base64_image, decode_image
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_registry() -> DocumentTypeRegistry:
    """Build and populate the document-type registry.

    Returns:
        A registry with all implemented document types registered.
    """
    registry = DocumentTypeRegistry()
    registry.register(DriverLicenseFront())
    registry.register(MyNumberCardFront())
    return registry


def run_infer(args: argparse.Namespace) -> int:
    """Run inference on a single image.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 = success).
    """
    with open(args.image, "rb") as f:
        image = decode_image(f.read())

    registry = build_registry()

    if args.document_type:
        doc_type = registry.get_by_id(args.document_type)
    else:
        doc_type = registry.detect(image)

    # Lazy-load heavy modules to keep CLI startup fast
    from src.detection.doctr_detector import DoctrDetector
    from src.pipeline.field_extractor import FieldExtractor
    from src.pipeline.ocr_engine import OCREngine
    from src.postprocessing.validator import Validator
    from src.recognition.trocr_recognizer import TrocrRecognizer

    device = args.device or "cuda"
    detector = DoctrDetector(device=device)
    recognizer = TrocrRecognizer(device=device, finetuned_path=args.model_path)
    extractor = FieldExtractor(doc_type=doc_type)
    validator = Validator()

    engine = OCREngine(
        doc_type=doc_type,
        detector=detector,
        recognizer=recognizer,
        extractor=extractor,
        validator=validator,
    )

    result = engine.process(image)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    """CLI entry point.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(prog="yomitori", description="Yomitori OCR engine")
    sub = parser.add_subparsers(dest="command")

    infer = sub.add_parser("infer", help="Run OCR on a single image")
    infer.add_argument("--image", required=True, help="Path to the input image")
    infer.add_argument(
        "--document_type",
        default=None,
        help="Document type ID (auto-detect if omitted)",
    )
    infer.add_argument("--device", default=None, help="Device: cuda or cpu")
    infer.add_argument("--model_path", default=None, help="Fine-tuned model path")
    infer.set_defaults(func=run_infer)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())