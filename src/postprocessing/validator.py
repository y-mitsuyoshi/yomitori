"""Generic validation engine for recognized fields."""

import re

from src.document_types.base import ValidationRule


class Validator:
    """Apply validation rules to recognized field values."""

    @staticmethod
    def validate(text: str, rule: ValidationRule) -> bool:
        """Validate a field value against a rule.

        Args:
            text: Recognized text value.
            rule: Validation rule to apply.

        Returns:
            True if validation passes (or rule has no constraints).
        """
        if rule.pattern:
            if not re.search(rule.pattern, text):
                return False
        if rule.check_digit:
            if not Validator._check_digit_ok(text):
                return False
        return True

    @staticmethod
    def _check_digit_ok(text: str) -> bool:
        """Placeholder for check-digit verification (e.g. MyNumber).

        Args:
            text: Numeric string.

        Returns:
            True if valid. Currently a pass-through (override per document).
        """
        return True

    @staticmethod
    def validate_all(
        fields: dict[str, dict],
        rules: dict[str, ValidationRule],
    ) -> dict[str, bool]:
        """Validate all fields against their rules.

        Args:
            fields: Field dict keyed by field name with ``value`` key.
            rules: Validation rules keyed by field name.

        Returns:
            Dict of field name → bool (validation passed).
        """
        results: dict[str, bool] = {}
        for name, data in fields.items():
            rule = rules.get(name)
            if rule is None:
                results[name] = True
                continue
            value = data.get("value", "")
            results[name] = Validator.validate(value, rule)
        return results