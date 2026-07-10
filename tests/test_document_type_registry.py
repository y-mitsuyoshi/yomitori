"""Tests for the document-type registry."""

import numpy as np

from src.document_types.driver_license import DriverLicenseFront
from src.document_types.registry import DocumentTypeRegistry


def test_registry_register_and_list():
    """Registry should list registered types."""
    reg = DocumentTypeRegistry()
    reg.register(DriverLicenseFront())
    assert "driver_license_front" in reg.list_types()
    assert len(reg) == 1


def test_registry_get_by_id():
    """get_by_id should return the correct type."""
    reg = DocumentTypeRegistry()
    reg.register(DriverLicenseFront())
    dt = reg.get_by_id("driver_license_front")
    assert dt.document_type_id == "driver_license_front"


def test_registry_get_by_id_unknown():
    """get_by_id should raise ValueError for unknown IDs."""
    import pytest

    reg = DocumentTypeRegistry()
    reg.register(DriverLicenseFront())
    with pytest.raises(ValueError):
        reg.get_by_id("nonexistent")


def test_registry_detect():
    """detect should return the best matching type."""
    reg = DocumentTypeRegistry()
    reg.register(DriverLicenseFront())
    # 1.585 aspect → 2400x1515
    img = np.ones((1515, 2400, 3), dtype=np.uint8)
    dt = reg.detect(img)
    assert dt.document_type_id == "driver_license_front"


def test_registry_empty():
    """detect should raise on empty registry."""
    import pytest

    reg = DocumentTypeRegistry()
    with pytest.raises(RuntimeError):
        reg.detect(np.ones((100, 100, 3), dtype=np.uint8))