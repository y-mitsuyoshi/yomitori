"""MyNumber card (front) document type implementation."""

from src.document_types.base import DocumentType, ValidationRule, Zone


class MyNumberCardFront(DocumentType):
    """マイナンバーカード（表面）— Japanese MyNumber card, front side."""

    @property
    def document_type_id(self) -> str:
        return "mynumber_card_front"

    @property
    def card_aspect_ratio(self) -> float:
        return 1.586  # 85.6 mm / 54.0 mm (ISO/IEC 7810 ID-1)

    @property
    def normalized_size(self) -> tuple[int, int]:
        return (2400, 1513)

    @property
    def zones(self) -> dict[str, Zone]:
        return {
            "name": Zone(
                "name", "氏名",
                0.05, 0.05, 0.65, 0.20,
                label_remove=True,
            ),
            "birth_date": Zone(
                "birth_date", "生年月日",
                0.05, 0.20, 0.65, 0.32,
                label_remove=True,
                date_format="japanese_era",
            ),
            "sex": Zone(
                "sex", "性別",
                0.05, 0.32, 0.30, 0.42,
                label_remove=True,
            ),
            "address": Zone(
                "address", "住所",
                0.05, 0.42, 0.75, 0.62,
                label_remove=True,
                normalize="fullwidth_to_halfwidth",
            ),
            "mynumber": Zone(
                "mynumber", "個人番号",
                0.05, 0.65, 0.55, 0.80,
                label_remove=True,
                whitelist="0123456789",
            ),
            "validity": Zone(
                "validity", "有効期限",
                0.05, 0.80, 0.55, 0.92,
                label_remove=True,
                date_format="japanese_era",
            ),
            "issuer": Zone(
                "issuer", "発行",
                0.40, 0.80, 0.90, 0.92,
                label_remove=True,
            ),
        }

    @property
    def validation_rules(self) -> dict[str, ValidationRule]:
        return {
            "mynumber": ValidationRule(
                pattern=r"^[0-9]{12}$",
                description="12桁のマイナンバー",
            ),
            "birth_date": ValidationRule(
                pattern=r"^(昭和|平成|令和)[0-9]+年[0-9]+月[0-9]+日",
                description="和暦生年月日",
            ),
            "validity": ValidationRule(
                pattern=r"^(昭和|平成|令和)[0-9]+年[0-9]+月[0-9]+日",
                description="和暦有効期限",
            ),
        }

    def detect(self, image: "np.ndarray") -> float:
        """Score the image as a MyNumber card.

        Uses aspect-ratio matching. MyNumber cards have a very similar
        aspect ratio to driver's licenses (both ID-1 size), so this
        score alone may not distinguish them — color histogram or
        layout features should be added for production use.

        Args:
            image: Input image (BGR ndarray).

        Returns:
            Score in [0.0, 1.0].
        """
        import numpy as np  # noqa: F401

        score = self._check_aspect_ratio(image, self.card_aspect_ratio)
        return score