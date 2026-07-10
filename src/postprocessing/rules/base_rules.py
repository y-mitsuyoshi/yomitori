"""Base class for document-specific validation rule plugins."""

from abc import ABC, abstractmethod

from src.document_types.base import ValidationRule


class BaseRules(ABC):
    """Abstract base for document-specific rule plugins.

    New document types subclass this to implement custom validation
    (check digits, cross-field consistency, etc.) beyond regex patterns.
    """

    @property
    @abstractmethod
    def document_type_id(self) -> str:
        """Document type ID this ruleset applies to."""
        ...

    @abstractmethod
    def get_rules(self) -> dict[str, ValidationRule]:
        """Return the validation rules dict.

        Returns:
            Mapping of field name → ValidationRule.
        """
        ...

    def extra_validate(self, fields: dict) -> list[str]:
        """Hook for cross-field or custom validation.

        Args:
            fields: Recognized fields dict.

        Returns:
            List of warning messages (empty if all OK).
        """
        return []