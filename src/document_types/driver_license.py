"""Driver's license (front) document type implementation."""

from src.document_types.base import DocumentType, ValidationRule, Zone


class DriverLicenseFront(DocumentType):
    """運転免許証（表面）— Japanese driver's license, front side."""

    @property
    def document_type_id(self) -> str:
        return "driver_license_front"

    @property
    def card_aspect_ratio(self) -> float:
        return 1.585  # 8.56 cm / 5.4 cm

    @property
    def normalized_size(self) -> tuple[int, int]:
        return (2400, 1512)

    @property
    def zones(self) -> dict[str, Zone]:
        return {
            "name": Zone(
                "name", "氏名",
                0.00, 0.00, 0.55, 0.12,
                label_remove=True,
            ),
            "birth_date": Zone(
                "birth_date", "生年月日",
                0.45, 0.00, 0.80, 0.12,
                label_remove=True,
                date_format="japanese_era",
            ),
            "address": Zone(
                "address", "住所",
                0.00, 0.12, 0.70, 0.26,
                label_remove=True,
                normalize="fullwidth_to_halfwidth",
            ),
            "issue_date": Zone(
                "issue_date", "交付",
                0.00, 0.26, 0.70, 0.33,
                label_remove=True,
                date_format="japanese_era",
            ),
            "expiry_date": Zone(
                "expiry_date", "有効期限",
                0.00, 0.33, 0.70, 0.40,
                label_remove=True,
                date_format="japanese_era",
            ),
            "conditions": Zone(
                "conditions", "条件等",
                0.00, 0.40, 0.70, 0.56,
                label_remove=True,
            ),
            "license_number": Zone(
                "license_number", "免許証番号",
                0.00, 0.60, 0.70, 0.78,
                label_remove=True,
                whitelist="0123456789",
            ),
            "license_type": Zone(
                "license_type", "免許種類",
                0.00, 0.78, 0.70, 0.90,
                label_remove=True,
            ),
        }

    @property
    def validation_rules(self) -> dict[str, ValidationRule]:
        return {
            "license_number": ValidationRule(
                pattern=r"^[0-9]{12}$",
                description="12桁の数字",
            ),
            "birth_date": ValidationRule(
                pattern=r"^(昭和|平成|令和)[0-9]+年[0-9]+月[0-9]+日",
                description="和暦生年月日",
            ),
            "expiry_date": ValidationRule(
                pattern=r"^(昭和|平成|令和)[0-9]+年[0-9]+月[0-9]+日",
                description="和暦有効期限",
            ),
        }

    def detect(self, image: "np.ndarray") -> float:
        """Score the image as a driver's license.

        Currently uses only aspect-ratio. Future: color histogram, photo area.

        Args:
            image: Input image (BGR ndarray).

        Returns:
            Score in [0.0, 1.0].
        """
        import numpy as np  # noqa: F401

        score = self._check_aspect_ratio(image, self.card_aspect_ratio)
        return score