"""Tests for the validator."""

import pytest

from src.document_types.base import ValidationRule
from src.postprocessing.validator import Validator


def test_validate_pattern_pass():
    """Should pass when pattern matches."""
    rule = ValidationRule(pattern=r"^[0-9]{12}$", description="12 digits")
    assert Validator.validate("010203040506", rule) is True


def test_validate_pattern_fail():
    """Should fail when pattern doesn't match."""
    rule = ValidationRule(pattern=r"^[0-9]{12}$", description="12 digits")
    assert Validator.validate("abc", rule) is False


def test_validate_no_rule_passes():
    """Should pass when no pattern or check_digit."""
    rule = ValidationRule()
    assert Validator.validate("anything", rule) is True


def test_validate_check_digit():
    """check_digit=True should pass (placeholder always returns True)."""
    rule = ValidationRule(check_digit=True)
    assert Validator.validate("123456789012", rule) is True


def test_validate_all():
    """validate_all should return per-field results."""
    fields = {"num": {"value": "010203040506"}, "name": {"value": "山田太郎"}}
    rules = {"num": ValidationRule(pattern=r"^[0-9]{12}$")}
    results = Validator.validate_all(fields, rules)
    assert results["num"] is True
    assert results["name"] is True  # no rule → True


def test_validate_all_missing_value():
    """validate_all should default to empty string when 'value' key missing."""
    fields = {"num": {}}
    rules = {"num": ValidationRule(pattern=r"^[0-9]{12}$")}
    results = Validator.validate_all(fields, rules)
    assert results["num"] is False  # "" doesn't match 12-digit pattern


def test_validate_all_no_rule_for_field():
    """validate_all should return True for fields without rules."""
    fields = {"unknown_field": {"value": "anything"}}
    rules = {}
    results = Validator.validate_all(fields, rules)
    assert results["unknown_field"] is True


def test_validate_pattern_and_check_digit_both():
    """Should pass both pattern and check_digit checks."""
    rule = ValidationRule(pattern=r"^[0-9]+$", check_digit=True)
    assert Validator.validate("12345", rule) is True