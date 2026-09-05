"""Configuration helpers shared by training and inference tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and require a mapping at the root."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}, got {type(config).__name__}")
    return config


def require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a required mapping section with a useful error message."""

    section = config.get(name)
    if not isinstance(section, dict):
        raise KeyError(f"Configuration section '{name}' is required")
    return section

