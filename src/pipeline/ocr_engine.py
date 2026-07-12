"""End-to-end OCR engine integrating preprocessing, detection, recognition, and postprocessing."""

from typing import Optional

import numpy as np

from src.detection.doctr_detector import DoctrDetector, DetectionResult
from src.document_types.base import DocumentType
from src.pipeline.field_extractor import FieldExtractor
from src.postprocessing.normalizer import (
    apply_whitelist,
    fullwidth_to_halfwidth,
    japanese_era_to_iso,
    remove_label,
)
from src.postprocessing.rules.base_rules import BaseRules
from src.postprocessing.rules.driver_license_rules import DriverLicenseRules
from src.postprocessing.validator import Validator
from src.preprocessing.homography import HomographyCorrector
from src.recognition.trocr_recognizer import TrocrRecognizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CONFIDENCE_THRESHOLD = 0.7

_RULES_REGISTRY: dict[str, BaseRules] = {
    "driver_license_front": DriverLicenseRules(),
}


class OCREngine:
    """Integrated OCR engine that runs the full pipeline for a document type.

    Args:
        doc_type: Document type plugin to use.
        detector: docTR detector instance.
        recognizer: TrOCR recognizer instance.
        extractor: Field extractor instance.
        validator: Validator instance.
        normalizer: Normalizer module (use default if None).
        confidence_threshold: Fields below this confidence are flagged low_confidence.
    """

    def __init__(
        self,
        doc_type: DocumentType,
        detector: DoctrDetector,
        recognizer: TrocrRecognizer,
        extractor: FieldExtractor,
        validator: Validator,
        confidence_threshold: float = _CONFIDENCE_THRESHOLD,
    ) -> None:
        self.doc_type = doc_type
        self.detector = detector
        self.recognizer = recognizer
        self.extractor = extractor
        self.validator = validator
        self.confidence_threshold = confidence_threshold

    def process(self, image: np.ndarray) -> dict:
        """Run the full OCR pipeline on an input image.

        Args:
            image: Input image (BGR ndarray).

        Returns:
            Structured result dict per the output format spec.
        """
        # 1. Preprocessing (homography correction)
        corrected, preprocessing_info = self._preprocess(image)

        # 2. Text detection (docTR)
        detections = self.detector.detect(corrected)

        # 3. Text recognition (TrOCR)
        recognized = self._recognize(detections)

        # 4. Field classification (zone-based)
        fields_raw = self._classify(recognized)

        # 5. Postprocessing (normalization + validation)
        result_fields, warnings = self._postprocess(fields_raw)

        # 6. Cross-field validation (extra_validate)
        cross_warnings = self._cross_validate(result_fields)
        warnings.extend(cross_warnings)

        # 7. Overall confidence score
        overall_confidence = self._compute_overall_confidence(result_fields)

        # 8. Status determination
        status = self._determine_status(result_fields, preprocessing_info, warnings)

        return {
            "status": status,
            "document_type": self.doc_type.document_type_id,
            "fields": result_fields,
            "overall_confidence": overall_confidence,
            "preprocessing": preprocessing_info,
            "warnings": warnings,
        }

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """Run homography correction.

        Args:
            image: Input image.

        Returns:
            Tuple of (corrected_image, info_dict).
        """
        corrector = HomographyCorrector(target_size=self.doc_type.normalized_size)
        return corrector.correct(image)

    def _recognize(self, detections: list[DetectionResult]) -> list[dict]:
        """Recognize text from detected line images.

        Args:
            detections: List of DetectionResult.

        Returns:
            List of dicts with text, confidence, bbox, center.
        """
        recognized: list[dict] = []
        if not detections:
            return recognized

        line_images = [d.line_image for d in detections]
        results = self.recognizer.recognize_batch(line_images)

        for det, (text, conf) in zip(detections, results):
            recognized.append(
                {
                    "text": text,
                    "confidence": conf,
                    "bbox": det.bbox,
                    "center": det.center_normalized,
                }
            )
        return recognized

    def _classify(self, recognized: list[dict]) -> dict[str, dict]:
        """Assign recognized lines to fields and merge multi-line fields.

        Args:
            recognized: List of recognized line dicts.

        Returns:
            Dict of field_name → {text, confidence}.
        """
        fields: dict[str, dict] = {}
        for rec in recognized:
            cx, cy = rec["center"]
            field_name = self.extractor.assign(cx, cy)
            if field_name is None:
                continue
            if field_name in fields:
                fields[field_name]["text"] += rec["text"]
                fields[field_name]["confidence"] = min(
                    fields[field_name]["confidence"], rec["confidence"]
                )
            else:
                fields[field_name] = {
                    "text": rec["text"],
                    "confidence": rec["confidence"],
                }
        return fields

    def _postprocess(
        self, fields_raw: dict[str, dict]
    ) -> tuple[dict[str, dict], list[str]]:
        """Apply normalization, label removal, date conversion, and validation.

        Args:
            fields_raw: Raw classified fields.

        Returns:
            Tuple of (result_fields, warnings).
        """
        result_fields: dict[str, dict] = {}
        warnings: list[str] = []

        for field_name, data in fields_raw.items():
            zone = self.doc_type.zones.get(field_name)
            text = data["text"]
            conf = data["confidence"]

            # Label removal
            if zone and zone.label_remove:
                text = remove_label(text, zone.label)

            # Fullwidth → halfwidth
            if zone and zone.normalize == "fullwidth_to_halfwidth":
                text = fullwidth_to_halfwidth(text)

            # Whitelist
            if zone and zone.whitelist:
                text = apply_whitelist(text, zone.whitelist)

            field_result: dict = {
                "value": text.strip(),
                "confidence": conf,
                "low_confidence": conf < self.confidence_threshold,
            }

            # Japanese era → ISO date conversion
            if zone and zone.date_format == "japanese_era":
                raw, iso = japanese_era_to_iso(text)
                field_result["raw"] = raw
                field_result["iso"] = iso
                field_result["value"] = iso or raw

            # Validation
            rule = self.doc_type.validation_rules.get(field_name)
            if rule:
                # Validate against the raw text (before ISO conversion) for era patterns
                validate_text = text if zone and zone.date_format == "japanese_era" else field_result["value"]
                passed = self.validator.validate(validate_text, rule)
                field_result["validation_passed"] = passed
                if not passed:
                    warnings.append(
                        f"{field_name}: validation failed - {rule.description}"
                    )

            result_fields[field_name] = field_result

        return result_fields, warnings

    def _cross_validate(self, result_fields: dict[str, dict]) -> list[str]:
        """Run cross-field validation rules if available for this document type.

        Args:
            result_fields: Processed fields.

        Returns:
            List of cross-field validation warnings.
        """
        rules_plugin = _RULES_REGISTRY.get(self.doc_type.document_type_id)
        if rules_plugin is None:
            return []
        return rules_plugin.extra_validate(result_fields)

    @staticmethod
    def _compute_overall_confidence(result_fields: dict[str, dict]) -> float:
        """Compute the mean confidence across all fields.

        Args:
            result_fields: Processed fields.

        Returns:
            Average confidence, or 0.0 if no fields.
        """
        if not result_fields:
            return 0.0
        total = sum(f.get("confidence", 0.0) for f in result_fields.values())
        return round(total / len(result_fields), 4)

    def _determine_status(
        self, result_fields: dict[str, dict], preprocessing_info: dict,
        warnings: list[str] | None = None,
    ) -> str:
        """Determine the overall processing status.

        Args:
            result_fields: Processed fields.
            preprocessing_info: Preprocessing metadata.
            warnings: Accumulated warnings (affects status if non-empty from cross-field).

        Returns:
            One of ``"success"``, ``"partial"``, ``"failed"``.
        """
        if preprocessing_info.get("fallback_used", False) and not result_fields:
            return "failed"
        if not result_fields:
            return "failed"

        has_low_conf = any(f.get("low_confidence", False) for f in result_fields.values())
        has_validation_fail = any(
            not f.get("validation_passed", True) for f in result_fields.values()
        )
        if has_low_conf or has_validation_fail:
            return "partial"
        return "success"