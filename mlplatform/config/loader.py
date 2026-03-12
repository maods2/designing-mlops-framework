from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*; returns a new dict."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result



def _load_config_profiles(
    profile_names: list[str],
    config_dir: Path | None,
) -> dict[str, Any]:
    """Load and merge config profile YAML files in order."""
    if not config_dir or not profile_names:
        return {}
    merged: dict[str, Any] = {}
    for name in profile_names:
        cfg_path = config_dir / f"{name}.yaml"
        if cfg_path.exists():
            profile_data = _load_yaml(cfg_path)
            merged = _deep_merge(merged, profile_data)
    return merged


def load_config_profiles(
    profile_names: list[str],
    config_dir: str | Path,
) -> dict[str, Any]:
    """Load and merge config profile YAML files in order.

    Profiles are loaded from ``config_dir/{name}.yaml`` and deep-merged.
    Later profiles override earlier ones.

    Example::

        merged = load_config_profiles(
            ["global", "dev"],
            config_dir="my_model/config",
        )
        cfg = TrainingConfig(merged)
    """
    path = Path(config_dir) if isinstance(config_dir, str) else config_dir
    return _load_config_profiles(profile_names, path)
