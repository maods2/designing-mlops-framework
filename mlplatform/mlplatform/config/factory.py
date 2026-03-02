"""Factory-based configuration loader for the new flat YAML schema.

Load order (later overrides earlier):
  1. Global: config/global.yaml (always)
  2. Pipeline-specific: config/{pipeline_type}-local.yaml (e.g. train-local, predict-local)
  3. Task-level: task entry from tasks list
  4. Profiles: config/<profile>.yaml for each name in task's config array (or CLI override)

Profiles are resolved at execution time when a task is selected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mlplatform.config.schema import TaskConfig, UnifiedPipelineConfig

PipelineType = str  # "training" | "prediction" | extensible


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


def _find_config_dir(pipeline_path: Path) -> Path | None:
    """Search for a config/ directory near the pipeline file.

    Search order: sibling of pipeline dir, then parent, then grandparent.
    For a pipeline at ``my_model/pipeline/train.yaml`` this finds
    ``my_model/config/`` first (recommended layout).
    """
    for parent in [
        pipeline_path.parent,
        pipeline_path.parent.parent,
        pipeline_path.parent.parent.parent,
    ]:
        candidate = parent / "config"
        if candidate.is_dir():
            return candidate
    return None


def _load_config_profile(profile_name: str, config_dir: Path | None) -> dict[str, Any]:
    """Load a single config profile YAML file."""
    if not config_dir:
        return {}
    cfg_path = config_dir / f"{profile_name}.yaml"
    if cfg_path.exists():
        return _load_yaml(cfg_path)
    return {}


def _load_config_profiles(
    profile_names: list[str],
    config_dir: Path | None,
) -> dict[str, Any]:
    """Load and merge config profile YAML files in order."""
    if not config_dir or not profile_names:
        return {}
    merged: dict[str, Any] = {}
    for name in profile_names:
        profile_data = _load_config_profile(name, config_dir)
        merged = _deep_merge(merged, profile_data)
    return merged


# Keys that map directly to TaskConfig/ModelConfig fields (excluding optional_configs)
_MODEL_CONFIG_KEYS = frozenset({
    "model_name", "module", "compute", "training_platform", "platform",
    "prediction_dataset_name", "prediction_table_name", "model_id", "model_version",
    "prediction_output_dataset_table", "predicted_label_column_name",
    "predicted_timestamp_column_name", "predicted_probability_column_name",
    "unique_identifier_column", "input_path", "output_path",
})

# Orchestration keys stored on TaskConfig
_ORCHESTRATION_KEYS = frozenset({
    "task_id", "task_key", "task_type", "config", "environment_key",
    "condition_task", "depends_on", "spark_python_task",
})


def _parse_task_config(entry: dict[str, Any]) -> TaskConfig:
    """Build a TaskConfig from a merged task entry.

    The entry is the result of: global + pipeline-specific + task + profiles.
    Extra top-level keys go into optional_configs.
    """
    platform = (
        entry.get("training_platform")
        or entry.get("platform")
        or entry.get("serving_platform")
        or "VertexAI"
    )

    optional = dict(entry.get("optional_configs") or {})
    for key, val in entry.items():
        if key not in _MODEL_CONFIG_KEYS and key not in _ORCHESTRATION_KEYS:
            if key not in optional:
                optional[key] = val

    return TaskConfig(
        task_id=entry.get("task_id") or entry.get("task_key", "default"),
        task_type=entry.get("task_type", "training"),
        model_name=entry.get("model_name") or entry.get("task_id", ""),
        module=entry.get("module", ""),
        compute=entry.get("compute", "s"),
        platform=platform,
        optional_configs=optional,
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
        condition_task=entry.get("condition_task"),
        depends_on=entry.get("depends_on"),
    )


def _resolve_profile_names(
    task_entry: dict[str, Any],
    config_names: list[str] | None,
) -> list[str]:
    """Resolve profile names from task config or CLI override."""
    if config_names is not None:
        return config_names
    raw_cfg = task_entry.get("config") or []
    if isinstance(raw_cfg, str):
        return [p.strip() for p in raw_cfg.split(",") if p.strip()]
    return list(raw_cfg)


def _pipeline_type_to_config_name(pipeline_type: str) -> str:
    """Map pipeline_type to pipeline-specific config file name."""
    mapping = {
        "training": "train-local",
        "prediction": "predict-local",
    }
    return mapping.get(pipeline_type, f"{pipeline_type}-local")


class PipelineConfigLoader:
    """Loads pipeline config using the four-layer merge strategy."""

    def load(
        self,
        pipeline_path: str | Path,
        task_id: str | None = None,
        config_names: list[str] | None = None,
        config_dir: str | Path | None = None,
    ) -> UnifiedPipelineConfig:
        """Load and merge pipeline configuration.

        Args:
            pipeline_path: Path to the pipeline YAML file (e.g. train.yaml, predict.yaml).
            task_id: If provided, only resolve and include this task's config.
            config_names: CLI override for profile names (overrides task's config array).
            config_dir: Directory containing config profiles. Defaults to auto-detection.

        Returns:
            UnifiedPipelineConfig with tasks populated. When task_id is given,
            only that task is included with its fully merged config.
        """
        pipeline_path = Path(pipeline_path)
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {pipeline_path}")

        raw = _load_yaml(pipeline_path)

        resolved_config_dir: Path | None
        if config_dir is not None:
            resolved_config_dir = Path(config_dir)
        else:
            resolved_config_dir = _find_config_dir(pipeline_path)

        pipeline_type = raw.get("pipeline_type", "training")
        pipeline_specific_name = _pipeline_type_to_config_name(pipeline_type)

        # Layer 1: Global
        global_cfg = _load_config_profile("global", resolved_config_dir)

        # Layer 2: Pipeline-specific
        pipeline_specific_cfg = _load_config_profile(
            pipeline_specific_name, resolved_config_dir
        )

        # Base = global + pipeline-specific + pipeline root (root overrides)
        base_effective = _deep_merge(global_cfg, pipeline_specific_cfg)
        base_effective = _deep_merge(base_effective, raw)

        # When config_names (CLI override) is provided, merge those profiles into
        # pipeline-level base so log_level etc. reflect the merged result
        if config_names:
            cli_profile_data = _load_config_profiles(config_names, resolved_config_dir)
            base_effective = _deep_merge(base_effective, cli_profile_data)

        # Exclude tasks from base - we merge per-task
        base_for_tasks = {k: v for k, v in base_effective.items() if k != "tasks"}

        task_entries = raw.get("tasks", [])
        if not task_entries:
            raise ValueError(
                f"Pipeline {pipeline_path} has no 'tasks' key or empty tasks list"
            )

        all_config_profiles: list[str] = []
        resolved_tasks: list[TaskConfig] = []

        for task_entry in task_entries:
            tid = task_entry.get("task_id") or task_entry.get("task_key")
            if task_id is not None and tid != task_id:
                continue

            # Layer 3: Task-level (merge base + task entry)
            task_base = _deep_merge(base_for_tasks, task_entry)

            # Layer 4: Profiles (resolved at execution time)
            profile_names = _resolve_profile_names(task_entry, config_names)
            profile_data = _load_config_profiles(profile_names, resolved_config_dir)
            merged_entry = _deep_merge(task_base, profile_data)

            resolved_tasks.append(_parse_task_config(merged_entry))
            for p in profile_names:
                if p not in all_config_profiles:
                    all_config_profiles.append(p)

        if task_id is not None and not resolved_tasks:
            raise ValueError(
                f"Task '{task_id}' not found in pipeline {pipeline_path}. "
                f"Available tasks: {[t.get('task_id') or t.get('task_key') for t in task_entries]}"
            )

        return UnifiedPipelineConfig(
            pipeline_name=base_effective.get("pipeline_name", "default_pipeline"),
            pipeline_type=pipeline_type,
            feature_name=base_effective.get("feature_name", "default"),
            schedule=base_effective.get("schedule", {}),
            environments=base_effective.get("environments", {}),
            tasks=resolved_tasks,
            log_level=base_effective.get("log_level", "INFO"),
            config_profiles=all_config_profiles,
        )


class ConfigLoaderFactory:
    """Factory for creating pipeline config loaders."""

    @staticmethod
    def create_loader() -> PipelineConfigLoader:
        """Create a PipelineConfigLoader instance."""
        return PipelineConfigLoader()

    @staticmethod
    def load_pipeline_config(
        pipeline_path: str | Path,
        task_id: str | None = None,
        config_names: list[str] | None = None,
        config_dir: str | Path | None = None,
    ) -> UnifiedPipelineConfig:
        """Convenience method: load pipeline config in one call."""
        loader = ConfigLoaderFactory.create_loader()
        return loader.load(
            pipeline_path=pipeline_path,
            task_id=task_id,
            config_names=config_names,
            config_dir=config_dir,
        )
