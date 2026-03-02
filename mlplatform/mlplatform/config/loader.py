"""Configuration loading from YAML files.

Supports both the legacy ``models:`` format and the new format where model/task
config is embedded inside ``resources.jobs.deployment.tasks`` (Databricks-like).

New format structure
--------------------
Framework values (``workflow_name``, ``pipeline_type``, etc.) live at the top
level. Each entry in ``resources.jobs.deployment.tasks`` carries **both**
framework model params (``model_name``, ``module``, ``compute``, etc.) **and**
the Databricks task params (``spark_python_task``, ``condition_task``, etc.).
Config profiles are declared per-task via the ``config:`` key inside each task.

Config profile merging
----------------------
For each task that declares ``config: [global, dev]``:

1. Load ``<config_dir>/global.yaml`` then ``<config_dir>/dev.yaml``.
2. Deep-merge them in order (later keys override earlier ones).
3. Framework top-level YAML values overlay the merged profile data (YAML wins).

``config_dir`` defaults to ``config/`` relative to the DAG file's *parent
directory* (i.e. alongside the model package), with fallback search upward.

CLI override
------------
Pass ``config_names=["global", "local"]`` to override every task's ``config:``
key from the command line::

    mlplatform run --dag train.yaml --config global,local
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mlplatform.config.schema import ModelConfig, WorkflowConfig


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


def _find_config_dir(dag_path: Path) -> Path | None:
    """Search for a config/ directory near the DAG file.

    Search order: sibling of DAG dir, then parent, then grandparent.
    For a DAG at ``my_model/pipeline/train.yaml`` this finds
    ``my_model/config/`` first (recommended layout).
    """
    for parent in [dag_path.parent, dag_path.parent.parent, dag_path.parent.parent.parent]:
        candidate = parent / "config"
        if candidate.is_dir():
            return candidate
    return None


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


def _parse_model_config(entry: dict[str, Any]) -> ModelConfig:
    """Build a ModelConfig from a task or models-list entry."""
    platform = entry.get("training_platform") or entry.get("serving_platform") or "VertexAI"
    return ModelConfig(
        model_name=entry.get("model_name") or entry.get("task_key", "default"),
        module=entry.get("module", ""),
        compute=entry.get("compute", "s"),
        platform=platform,
        optional_configs=entry.get("optional_configs") or {},
        prediction_dataset_name=entry.get("prediction_dataset_name"),
        prediction_table_name=entry.get("prediction_table_name"),
        model_id=entry.get("model_id"),
        model_version=entry.get("model_version", "latest"),
        prediction_output_dataset_table=entry.get("prediction_output_dataset_table"),
        predicted_label_column_name=entry.get("predicted_label_column_name"),
        predicted_timestamp_column_name=entry.get("predicted_timestamp_column_name"),
        predicted_probability_column_name=entry.get("predicted_probability_column_name"),
        unique_identifier_column=entry.get("unique_identifier_column"),
        input_path=entry.get("input_path"),
        output_path=entry.get("output_path"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_workflow_config(
    dag_path: str | Path,
    config_names: list[str] | None = None,
    config_dir: str | Path | None = None,
) -> WorkflowConfig:
    """Load a workflow config from a DAG YAML file.

    Supports two formats:

    * **New format** — ``resources.jobs.deployment.tasks`` carries both
      framework model params and Databricks task params; ``config:`` is
      declared inside each task.
    * **Legacy format** — top-level ``models:`` list with framework params.

    Args:
        dag_path: Path to the DAG YAML file.
        config_names: Override every task's ``config:`` key.  Pass
            ``["global", "local"]`` to load those profiles for all tasks.
            ``None`` uses each task's own ``config:`` key.
        config_dir: Directory containing config profile YAML files.  Defaults
            to auto-detection (searches for ``config/`` near the DAG file).
    """
    dag_path = Path(dag_path)
    if not dag_path.exists():
        raise FileNotFoundError(f"DAG config not found: {dag_path}")

    raw = _load_yaml(dag_path)

    resolved_config_dir: Path | None
    if config_dir is not None:
        resolved_config_dir = Path(config_dir)
    else:
        resolved_config_dir = _find_config_dir(dag_path)

    # --- Determine task/model entries and config profiles ---
    all_config_profiles: list[str] = []
    models: list[ModelConfig] = []

    # New format: resources.jobs.deployment.tasks
    deployment = (
        raw.get("resources", {})
           .get("jobs", {})
           .get("deployment", {})
    )
    task_entries = deployment.get("tasks") if deployment else None

    # effective holds merged profile data + raw; DAG values win (raw is merged last)
    effective: dict[str, Any] = raw

    if task_entries:
        for task_entry in task_entries:
            # Skip orchestration-only tasks (no model_name / module)
            if not task_entry.get("model_name") and not task_entry.get("module"):
                continue

            # Resolve config profiles for this task
            task_profile_names: list[str]
            if config_names is not None:
                task_profile_names = config_names
            else:
                raw_cfg = task_entry.get("config") or []
                if isinstance(raw_cfg, str):
                    task_profile_names = [p.strip() for p in raw_cfg.split(",") if p.strip()]
                else:
                    task_profile_names = list(raw_cfg)

            profile_data = _load_config_profiles(task_profile_names, resolved_config_dir)
            # Overlay raw top-level values on the merged profiles (raw wins)
            effective = _deep_merge(profile_data, raw)

            models.append(_parse_model_config(task_entry))
            for p in task_profile_names:
                if p not in all_config_profiles:
                    all_config_profiles.append(p)
    else:
        # Legacy format: top-level models: list
        # Resolve config from top-level config: key or CLI override
        if config_names is not None:
            top_profile_names = config_names
        else:
            raw_cfg = raw.get("config") or []
            if isinstance(raw_cfg, str):
                top_profile_names = [p.strip() for p in raw_cfg.split(",") if p.strip()]
            else:
                top_profile_names = list(raw_cfg)

        profile_data = _load_config_profiles(top_profile_names, resolved_config_dir)
        effective = _deep_merge(profile_data, raw)
        all_config_profiles = top_profile_names

        for entry in effective.get("models", []):
            models.append(_parse_model_config(entry))

    # Top-level framework values — effective merges profile data with DAG YAML (DAG wins)
    pipeline_type = effective.get("pipeline_type", "training")

    return WorkflowConfig(
        workflow_name=effective.get("workflow_name", "default_workflow"),
        execution_mode=effective.get("execution_mode", "sequential"),
        pipeline_type=pipeline_type,
        feature_name=effective.get("feature_name", "default"),
        config_version=effective.get("config_version", 2),
        models=models,
        log_level=effective.get("log_level", "INFO"),
        config_profiles=all_config_profiles,
    )
