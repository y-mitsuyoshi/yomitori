"""Driver's license specific validation rules."""

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
                pattern=r"^(昭和|平成|令和)[0-9]+年[0-9]+月[0-9]+日",
                description="和暦有効期限",
            ),
        }

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
        expiry_iso = expiry.get("iso")
        if birth_iso and expiry_iso and birth_iso > expiry_iso:
            warnings.append("生年月日が有効期限より後になっています")
        return warnings