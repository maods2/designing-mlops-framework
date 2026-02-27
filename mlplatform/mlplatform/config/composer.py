"""Composable workflow config loader with profile/domain overlays and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _get_required_paths(conf: dict[str, Any]) -> list[str]:
    required = conf.get("required") or []
    return [p for p in required if isinstance(p, str)]


def _has_dotted_path(conf: dict[str, Any], dotted_path: str) -> bool:
    cur: Any = conf
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _validate_required(conf: dict[str, Any], required_paths: list[str]) -> None:
    missing = [p for p in required_paths if not _has_dotted_path(conf, p)]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")


def _validate_profile_contract(conf: dict[str, Any], profile_name: str | None) -> None:
    if profile_name != "prod":
        return
    prod_required = [
        "cloud.region",
        "cloud.service_account",
        "cloud.network",
        "cloud.scheduling.max_retries",
    ]
    _validate_required(conf, prod_required)


def compose_workflow_dict(
    dag_path: str | Path,
    config_profile: str | None = None,
    domain: str | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose workflow config from defaults/profile/domain overlays + DAG file.

    Precedence: defaults < profile < domain < dag < runtime_overrides.
    """
    dag_path = Path(dag_path)
    dag_data = _load_yaml(dag_path)
    project_root = dag_path.parent

    merged: dict[str, Any] = {}
    required_paths: list[str] = []

    defaults_path = project_root / "config" / "defaults.yaml"
    if defaults_path.exists():
        defaults_data = _load_yaml(defaults_path)
        merged = _deep_merge(merged, defaults_data)
        required_paths.extend(_get_required_paths(defaults_data))

    if config_profile:
        profile_path = project_root / "config" / "profiles" / f"{config_profile}.yaml"
        if profile_path.exists():
            profile_data = _load_yaml(profile_path)
            merged = _deep_merge(merged, profile_data)
            required_paths.extend(_get_required_paths(profile_data))

    if domain:
        domain_path = project_root / "config" / "domains" / f"{domain}.yaml"
        if domain_path.exists():
            domain_data = _load_yaml(domain_path)
            merged = _deep_merge(merged, domain_data)
            required_paths.extend(_get_required_paths(domain_data))

    merged = _deep_merge(merged, dag_data)

    if runtime_overrides:
        merged = _deep_merge(merged, runtime_overrides)

    _validate_required(merged, required_paths)
    _validate_profile_contract(merged, config_profile)

    return merged
