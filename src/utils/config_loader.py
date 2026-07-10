"""YAML configuration file loader."""

from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    logger.debug("Loaded config: %s", p)
    return data or {}


def load_engine_config(config_dir: str | Path | None = None) -> dict[str, Any]:
    """Load the engine-level config (engine.yaml).

    Args:
        config_dir: Directory containing engine.yaml. Defaults to the
            project-level ``configs/`` directory.

    Returns:
        Engine configuration dict.
    """
    base = Path(config_dir) if config_dir else _DEFAULT_CONFIGS_DIR
    return load_yaml(base / "engine.yaml")


def load_document_type_config(
    document_type_id: str,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load a document-type YAML config from ``configs/document_types/``.

    Args:
        document_type_id: Document type ID (e.g. ``driver_license_front``).
        config_dir: Base configs directory. Defaults to the project configs dir.

    Returns:
        Document type configuration dict.
    """
    base = Path(config_dir) if config_dir else _DEFAULT_CONFIGS_DIR
    path = base / "document_types" / f"{document_type_id}.yaml"
    return load_yaml(path)


def list_document_type_configs(config_dir: str | Path | None = None) -> list[Path]:
    """List all document-type YAML files.

    Args:
        config_dir: Base configs directory.

    Returns:
        Sorted list of YAML file paths.
    """
    base = Path(config_dir) if config_dir else _DEFAULT_CONFIGS_DIR
    dt_dir = base / "document_types"
    if not dt_dir.exists():
        return []
    return sorted(dt_dir.glob("*.yaml"))