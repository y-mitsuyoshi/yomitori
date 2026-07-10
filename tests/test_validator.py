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


def test_validate_all():
    """validate_all should return per-field results."""
    fields = {"num": {"value": "010203040506"}, "name": {"value": "山田太郎"}}
    rules = {"num": ValidationRule(pattern=r"^[0-9]{12}$")}
    results = Validator.validate_all(fields, rules)
    assert results["num"] is True
    assert results["name"] is True  # no rule → True