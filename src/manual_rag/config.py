"""Configuration loading.

All tunable values (paths, model names, hyperparameters) live in
``configs/config.yaml`` so experiments never require code edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass
class Config:
    """Typed wrapper around the YAML config with path resolution helpers."""

    raw: dict[str, Any] = field(default_factory=dict)
    root: Path = PROJECT_ROOT

    def section(self, name: str) -> dict[str, Any]:
        """Return one top-level section of the config (empty dict if missing)."""
        return self.raw.get(name, {})

    def path(self, key: str) -> Path:
        """Resolve a value from the ``paths`` section relative to the repo root."""
        value = self.section("paths")[key]
        p = Path(value)
        return p if p.is_absolute() else self.root / p


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML config file.

    Args:
        path: Optional path to a config file. Defaults to ``configs/config.yaml``.

    Returns:
        A :class:`Config` instance.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(raw=raw)
