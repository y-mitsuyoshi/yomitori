"""Tests for the config loader."""

import pytest

from src.utils.config_loader import (
    list_document_type_configs,
    load_document_type_config,
    load_engine_config,
    load_yaml,
)


def test_load_yaml_valid(tmp_path):
    """load_yaml should parse a valid YAML file."""
    f = tmp_path / "test.yaml"
    f.write_text("key: value\nlist:\n  - a\n  - b\n")
    data = load_yaml(f)
    assert data["key"] == "value"
    assert data["list"] == ["a", "b"]


def test_load_yaml_not_found(tmp_path):
    """load_yaml should raise FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nonexistent.yaml")


def test_load_yaml_empty(tmp_path):
    """load_yaml should return empty dict for empty file."""
    f = tmp_path / "empty.yaml"
    f.write_text("")
    data = load_yaml(f)
    assert data == {}


def test_load_engine_config(tmp_path):
    """load_engine_config should load engine.yaml."""
    config_dir = tmp_path
    (config_dir / "engine.yaml").write_text("device: cuda\n")
    data = load_engine_config(config_dir)
    assert data["device"] == "cuda"


def test_load_document_type_config(tmp_path):
    """load_document_type_config should load from document_types/ subdirectory."""
    config_dir = tmp_path
    (config_dir / "document_types").mkdir()
    (config_dir / "document_types" / "test.yaml").write_text("id: test\n")
    data = load_document_type_config("test", config_dir)
    assert data["id"] == "test"


def test_list_document_type_configs(tmp_path):
    """list_document_type_configs should list YAML files."""
    config_dir = tmp_path
    dt_dir = config_dir / "document_types"
    dt_dir.mkdir()
    (dt_dir / "a.yaml").write_text("id: a\n")
    (dt_dir / "b.yaml").write_text("id: b\n")
    files = list_document_type_configs(config_dir)
    assert len(files) == 2


def test_list_document_type_configs_empty(tmp_path):
    """list_document_type_configs should return [] when dir doesn't exist."""
    files = list_document_type_configs(tmp_path)
    assert files == []