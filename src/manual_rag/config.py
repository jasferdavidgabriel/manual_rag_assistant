"""Configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass
class Config:
    """Typed wrapper around the YAML config with path helpers."""

    raw: dict[str, Any] = field(default_factory=dict)
    root: Path = PROJECT_ROOT

    def section(self, name: str) -> dict[str, Any]:
        """Return one top level section of the config."""
        return self.raw.get(name, {})

    def path(self, key: str) -> Path:
        """Resolve a value from the paths section relative to the repo root."""
        p = Path(self.section("paths")[key])
        return p if p.is_absolute() else self.root / p


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML config file (defaults to configs/config.yaml)."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        return Config(raw=yaml.safe_load(f) or {})
