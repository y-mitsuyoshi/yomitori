"""Driver's license specific validation rules."""

import re

from src.document_types.base import ValidationRule
from src.postprocessing.rules.base_rules import BaseRules


class DriverLicenseRules(BaseRules):
    """Validation rules for the driver's license (front) document."""

    @property
    def document_type_id(self) -> str:
        return "driver_license_front"

    def get_rules(self) -> dict[str, ValidationRule]:
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
                pattern=r"^[0-9]{4}年[0-9]+月[0-9]+日",
                description="西暦有効期限",
            ),
        }

    @staticmethod
    def _expiry_to_iso(value: str) -> str | None:
        """西暦有効期限文字列をISO日付に変換する。

        Args:
            value: "2027年12月31日" 形式の文字列。

        Returns:
            "2027-12-31" 形式のISO日付、または None。
        """
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", value)
        if not m:
            return None
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    def extra_validate(self, fields: dict) -> list[str]:
        """Cross-field validation for driver's license.

        Args:
            fields: Recognized fields.

        Returns:
            List of warning strings.
        """
        warnings: list[str] = []
        # Ensure expiry date is after birth date if both present
        birth = fields.get("birth_date", {})
        expiry = fields.get("expiry_date", {})
        birth_iso = birth.get("iso")
        expiry_iso = expiry.get("iso") or self._expiry_to_iso(expiry.get("value", ""))
        if birth_iso and expiry_iso and birth_iso > expiry_iso:
            warnings.append("生年月日が有効期限より後になっています")
        return warnings