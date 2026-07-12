"""Tests for base rules and driver license rules."""

from src.postprocessing.rules.base_rules import BaseRules
from src.postprocessing.rules.driver_license_rules import DriverLicenseRules
from src.document_types.base import ValidationRule


def test_driver_license_rules_document_type_id():
    """DriverLicenseRules should have the correct document_type_id."""
    rules = DriverLicenseRules()
    assert rules.document_type_id == "driver_license_front"


def test_driver_license_rules_get_rules():
    """get_rules should return validation rules dict."""
    rules = DriverLicenseRules()
    rule_dict = rules.get_rules()
    assert "license_number" in rule_dict
    assert isinstance(rule_dict["license_number"], ValidationRule)
    assert "birth_date" in rule_dict
    assert "expiry_date" in rule_dict


def test_extra_validate_both_missing():
    """extra_validate should return no warnings when fields missing."""
    rules = DriverLicenseRules()
    warnings = rules.extra_validate({})
    assert warnings == []


def test_extra_validate_birth_missing():
    """extra_validate should return no warnings when only one date present."""
    rules = DriverLicenseRules()
    warnings = rules.extra_validate({"birth_date": {"iso": "1990-01-01"}})
    assert warnings == []


def test_extra_validate_valid_order():
    """extra_validate should return no warnings when birth < expiry."""
    rules = DriverLicenseRules()
    warnings = rules.extra_validate({
        "birth_date": {"iso": "1990-01-01"},
        "expiry_date": {"iso": "2030-01-01"},
    })
    assert warnings == []


def test_extra_validate_inverted_order():
    """extra_validate should warn when birth > expiry."""
    rules = DriverLicenseRules()
    warnings = rules.extra_validate({
        "birth_date": {"iso": "2030-01-01"},
        "expiry_date": {"iso": "1990-01-01"},
    })
    assert len(warnings) == 1
    assert "生年月日が有効期限より後" in warnings[0]


def test_base_rules_extra_validate_default():
    """BaseRules.extra_validate default should return empty list."""

    class DummyRules(BaseRules):
        @property
        def document_type_id(self) -> str:
            return "dummy"

        def get_rules(self) -> dict[str, ValidationRule]:
            return {}

    rules = DummyRules()
    assert rules.extra_validate({}) == []